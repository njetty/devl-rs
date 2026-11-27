//! Bounded outbound event queue for one stream.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use crate::message::ServerEvent;

/// Outbound event queue with a fixed structural bound.
///
/// At most one pending `Partial` (a newer one replaces an older one) and one
/// pending `Credit` (grants are summed) are retained, so the queue never holds
/// more than `Started`, `Credit`, `Partial`, and one terminal event. A slow
/// reader therefore cannot make the server buffer without limit, and it never
/// blocks a worker.
pub(crate) struct EventQueue {
    inner: Mutex<VecDeque<ServerEvent>>,
    ready: Condvar,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl EventQueue {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
        }
    }

    /// Queue an event that the stream contract already accepted.
    pub(crate) fn push(&self, event: ServerEvent) {
        let mut q = lock(&self.inner);
        match event {
            ServerEvent::Partial { .. } => {
                q.retain(|e| !matches!(e, ServerEvent::Partial { .. }));
                q.push_back(event);
            }
            ServerEvent::Credit { samples } => {
                let merged = q.iter_mut().find_map(|e| match e {
                    ServerEvent::Credit { samples: held } => Some(held),
                    _ => None,
                });
                match merged {
                    Some(held) => *held += samples,
                    None => q.push_back(ServerEvent::Credit { samples }),
                }
            }
            other => {
                // A terminal event supersedes any unread partial.
                if matches!(
                    other,
                    ServerEvent::Final { .. } | ServerEvent::Cancelled | ServerEvent::Error(_)
                ) {
                    q.retain(|e| !matches!(e, ServerEvent::Partial { .. }));
                }
                q.push_back(other);
            }
        }
        self.ready.notify_all();
    }

    pub(crate) fn try_pop(&self) -> Option<ServerEvent> {
        lock(&self.inner).pop_front()
    }

    pub(crate) fn pop_timeout(&self, timeout: Duration) -> Option<ServerEvent> {
        let deadline = Instant::now() + timeout;
        let mut q = lock(&self.inner);
        loop {
            if let Some(e) = q.pop_front() {
                return Some(e);
            }
            let remaining = deadline.checked_duration_since(Instant::now())?;
            q = self
                .ready
                .wait_timeout(q, remaining)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        lock(&self.inner).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesces_partials_and_credit() {
        let q = EventQueue::new();
        for rev in 0..100 {
            q.push(ServerEvent::Partial {
                revision: rev,
                audio_samples: rev,
                text: String::new(),
            });
            q.push(ServerEvent::Credit { samples: 1 });
        }
        assert_eq!(q.len(), 2);
        let mut seen = Vec::new();
        while let Some(e) = q.try_pop() {
            seen.push(e);
        }
        assert!(seen.contains(&ServerEvent::Credit { samples: 100 }));
        assert!(
            seen.iter()
                .any(|e| matches!(e, ServerEvent::Partial { revision: 99, .. }))
        );
    }

    #[test]
    fn pop_timeout_returns_none_when_empty() {
        let q = EventQueue::new();
        assert!(q.pop_timeout(Duration::from_millis(1)).is_none());
    }
}
