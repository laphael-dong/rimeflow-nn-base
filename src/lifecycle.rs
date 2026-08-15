//! Thread-safe, one-shot Native/Web initialization lifecycle.

use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Condvar, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

use crate::backend::{
    BackendFactory, BackendInitRequest, ModelInput, RawModelOutput, ResolvedBackend, RuntimeBackend,
};
use crate::error::{InferenceError, InitFailure, InitializationStage};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum InitOutcome {
    Ready { resolved: ResolvedBackend },
    UseWebFallback { failure: InitFailure },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum WebInitOutcome {
    Ready { resolved: ResolvedBackend },
    TerminalFailure { failure: InitFailure },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleSnapshot {
    Uninitialized,
    InitializingNative,
    UseWebFallback,
    InitializingWeb,
    ReadyNative,
    ReadyWeb,
    WebTerminalFailure,
    Released,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleError {
    pub code: &'static str,
    pub message: String,
}

impl LifecycleError {
    fn released() -> Self {
        Self {
            code: "runtime_released",
            message: "released runtime cannot be initialized or used again".to_owned(),
        }
    }

    fn backend_already_ready() -> Self {
        Self {
            code: "backend_already_ready",
            message: "a backend is already fixed for this runtime".to_owned(),
        }
    }
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for LifecycleError {}

enum ReadyOrigin {
    Native,
    Web,
}

enum State<B> {
    Uninitialized,
    InitializingNative,
    NativeFallback(InitFailure),
    InitializingWeb,
    Ready {
        origin: ReadyOrigin,
        backend: B,
        resolved: ResolvedBackend,
    },
    WebTerminal(InitFailure),
    Released,
}

struct Inner<B> {
    state: State<B>,
    published_instances: usize,
    web_fallbacks: usize,
}

/// A runtime publishes at most one backend instance and never leaves Released.
pub struct RuntimeLifecycle<B> {
    inner: Mutex<Inner<B>>,
    changed: Condvar,
}

impl<B> Default for RuntimeLifecycle<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B> RuntimeLifecycle<B> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                state: State::Uninitialized,
                published_instances: 0,
                web_fallbacks: 0,
            }),
            changed: Condvar::new(),
        }
    }

    pub fn initialize_native<F>(
        &self,
        request: &BackendInitRequest,
        factory: &F,
    ) -> Result<InitOutcome, LifecycleError>
    where
        F: BackendFactory<B>,
    {
        let mut inner = self.lock_inner();
        loop {
            match &inner.state {
                State::Uninitialized => {
                    inner.state = State::InitializingNative;
                    break;
                }
                State::InitializingNative | State::InitializingWeb => {
                    inner = self.wait(inner);
                }
                State::Ready { resolved, .. } => {
                    return Ok(InitOutcome::Ready {
                        resolved: resolved.clone(),
                    });
                }
                State::NativeFallback(failure) | State::WebTerminal(failure) => {
                    return Ok(InitOutcome::UseWebFallback {
                        failure: failure.clone(),
                    });
                }
                State::Released => return Err(LifecycleError::released()),
            }
        }
        drop(inner);

        let result =
            catch_unwind(AssertUnwindSafe(|| factory.create(request))).unwrap_or_else(|_| {
                Err(InitFailure::new(
                    "native_factory_panicked",
                    InitializationStage::RuntimeLoad,
                    "backend factory panicked during initialization",
                ))
            });

        let mut inner = self.lock_inner();
        let outcome = match result {
            Ok(instance) => {
                inner.published_instances += 1;
                let resolved = instance.resolved;
                inner.state = State::Ready {
                    origin: ReadyOrigin::Native,
                    backend: instance.backend,
                    resolved: resolved.clone(),
                };
                InitOutcome::Ready { resolved }
            }
            Err(failure) => {
                inner.web_fallbacks += 1;
                inner.state = State::NativeFallback(failure.clone());
                InitOutcome::UseWebFallback { failure }
            }
        };
        self.changed.notify_all();
        Ok(outcome)
    }

    pub fn initialize_web<F>(
        &self,
        request: &BackendInitRequest,
        factory: &F,
    ) -> Result<WebInitOutcome, LifecycleError>
    where
        F: BackendFactory<B>,
    {
        let mut inner = self.lock_inner();
        loop {
            match &inner.state {
                State::Uninitialized | State::NativeFallback(_) => {
                    inner.state = State::InitializingWeb;
                    break;
                }
                State::InitializingNative | State::InitializingWeb => {
                    inner = self.wait(inner);
                }
                State::Ready {
                    origin: ReadyOrigin::Web,
                    resolved,
                    ..
                } => {
                    return Ok(WebInitOutcome::Ready {
                        resolved: resolved.clone(),
                    });
                }
                State::Ready {
                    origin: ReadyOrigin::Native,
                    ..
                } => return Err(LifecycleError::backend_already_ready()),
                State::WebTerminal(failure) => {
                    return Ok(WebInitOutcome::TerminalFailure {
                        failure: failure.clone(),
                    });
                }
                State::Released => return Err(LifecycleError::released()),
            }
        }
        drop(inner);

        let result =
            catch_unwind(AssertUnwindSafe(|| factory.create(request))).unwrap_or_else(|_| {
                Err(InitFailure::new(
                    "web_factory_panicked",
                    InitializationStage::RuntimeLoad,
                    "Web backend factory panicked during initialization",
                ))
            });

        let mut inner = self.lock_inner();
        let outcome = match result {
            Ok(instance) => {
                inner.published_instances += 1;
                let resolved = instance.resolved;
                inner.state = State::Ready {
                    origin: ReadyOrigin::Web,
                    backend: instance.backend,
                    resolved: resolved.clone(),
                };
                WebInitOutcome::Ready { resolved }
            }
            Err(failure) => {
                inner.state = State::WebTerminal(failure.clone());
                WebInitOutcome::TerminalFailure { failure }
            }
        };
        self.changed.notify_all();
        Ok(outcome)
    }

    pub fn release(&self) -> Result<(), LifecycleError> {
        let mut inner = self.lock_inner();
        while matches!(
            inner.state,
            State::InitializingNative | State::InitializingWeb
        ) {
            inner = self.wait(inner);
        }
        inner.state = State::Released;
        self.changed.notify_all();
        Ok(())
    }

    pub fn snapshot(&self) -> LifecycleSnapshot {
        match &self.lock_inner().state {
            State::Uninitialized => LifecycleSnapshot::Uninitialized,
            State::InitializingNative => LifecycleSnapshot::InitializingNative,
            State::NativeFallback(_) => LifecycleSnapshot::UseWebFallback,
            State::InitializingWeb => LifecycleSnapshot::InitializingWeb,
            State::Ready {
                origin: ReadyOrigin::Native,
                ..
            } => LifecycleSnapshot::ReadyNative,
            State::Ready {
                origin: ReadyOrigin::Web,
                ..
            } => LifecycleSnapshot::ReadyWeb,
            State::WebTerminal(_) => LifecycleSnapshot::WebTerminalFailure,
            State::Released => LifecycleSnapshot::Released,
        }
    }

    pub fn diagnostics(&self) -> Option<ResolvedBackend> {
        match &self.lock_inner().state {
            State::Ready { resolved, .. } => Some(resolved.clone()),
            _ => None,
        }
    }

    pub fn published_instance_count(&self) -> usize {
        self.lock_inner().published_instances
    }

    pub fn web_fallback_count(&self) -> usize {
        self.lock_inner().web_fallbacks
    }

    fn lock_inner(&self) -> MutexGuard<'_, Inner<B>> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn wait<'a>(&self, guard: MutexGuard<'a, Inner<B>>) -> MutexGuard<'a, Inner<B>> {
        self.changed
            .wait(guard)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl<B: RuntimeBackend> RuntimeLifecycle<B> {
    pub fn infer(&self, input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        input.validate()?;
        let mut inner = self.lock_inner();
        match &mut inner.state {
            State::Ready { backend, .. } => backend.infer(input),
            State::Released => Err(InferenceError::new(
                "runtime_released",
                "released runtime cannot run inference",
            )),
            _ => Err(InferenceError::new(
                "runtime_not_ready",
                "runtime has no published backend",
            )),
        }
    }
}
