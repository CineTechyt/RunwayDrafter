//! The live cloth simulation, off the user interface's back.
//!
//! A simulation step of a garment costs tens of milliseconds, more on a
//! laptop. Stepped on the interface's thread, every step is a frame the
//! view cannot draw, and the editor stutters while it drapes. Here the
//! solver runs on a thread of its own: the view draws at its own rate and
//! shows the newest cloth whenever a step is done. A slow machine drapes in
//! slower motion instead of in jerks.
//!
//! The simulation stops by itself once the cloth has settled: when hardly
//! any of it still moves from one step to the next.
//!
//! For headless use (screenshots, tests) the same work runs inline, step
//! for step with the frames, so results do not depend on timing.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

use cad_core::PatternDocument;
use glam::Vec3;
use topology_sync::{
    ClothSolver, Collider, DistanceField, GarmentMesh, SolverContext, SolverSettings, SolverStats,
};

/// A step moves a vertex less than this when it is at rest, metres.
const AT_REST: f32 = 0.0001;
/// Settled: no more than this share of the vertices still moves…
const STILL_MOVING: f32 = 0.01;
/// …for this many steps in a row…
const SETTLED_STEPS: u32 = 15;
/// …and not before the seams have had time to close.
const MIN_STEPS: u32 = 45;

/// What a step needs besides the cloth: owned, so it can travel to the
/// simulation thread.
#[derive(Clone)]
pub struct Scene {
    pub document: PatternDocument,
    pub settings: SolverSettings,
    pub colliders: Vec<Collider>,
    pub body: Option<Arc<DistanceField>>,
    pub body_mesh: Option<Arc<physics_collision::TriangleMesh>>,
    pub dt: f32,
}

impl Scene {
    fn context(&self) -> SolverContext<'_> {
        SolverContext {
            document: &self.document,
            settings: &self.settings,
            colliders: &self.colliders,
            body: self.body.as_ref(),
            body_mesh: self.body_mesh.as_ref(),
            ground: 0.0,
            dt: self.dt,
        }
    }
}

enum Command {
    /// New cloth: start over from `mesh` at rest.
    Rebind(Box<GarmentMesh>, Box<Scene>, u64),
    /// Fabric or grain changed; keep the motion.
    Refresh(Box<Scene>),
    /// The body, settings or document changed; keep the motion.
    Scene(Box<Scene>),
    /// The cloth was moved from outside: continue from here, at rest.
    Positions(Vec<Vec3>, u64),
    Grab(Option<(u32, Vec3, f32, f32)>),
    /// Keep stepping (`true`) or stop.
    Run(bool),
}

/// What the simulation reports after a step.
pub struct Progress {
    /// Which cloth this is: positions from before a rebind or an outside
    /// move are stale.
    pub epoch: u64,
    pub positions: Vec<Vec3>,
    /// Steps done since the last report.
    pub steps: u32,
    pub moved: bool,
    pub stats: SolverStats,
    /// The cloth has come to rest; the simulation stopped.
    pub settled: bool,
}

/// The solver and its own copy of the cloth.
struct Worker {
    solver: Box<dyn ClothSolver + Send>,
    mesh: GarmentMesh,
    scene: Option<Scene>,
    epoch: u64,
    running: bool,
    grabbing: bool,
    since_rebind: u32,
    still: u32,
}

impl Worker {
    fn handle(&mut self, command: Command) {
        match command {
            Command::Rebind(mesh, scene, epoch) => {
                self.mesh = *mesh;
                self.solver.rebind(&self.mesh, &scene.context());
                self.scene = Some(*scene);
                self.epoch = epoch;
                self.since_rebind = 0;
                self.still = 0;
            }
            Command::Refresh(scene) => {
                self.solver.refresh_materials(&self.mesh, &scene.context());
                self.scene = Some(*scene);
            }
            Command::Scene(scene) => self.scene = Some(*scene),
            Command::Positions(positions, epoch) => {
                if positions.len() == self.mesh.positions.len() {
                    self.mesh.positions = positions;
                    self.solver.set_positions(&self.mesh);
                }
                self.epoch = epoch;
                self.still = 0;
            }
            Command::Grab(grab) => {
                self.grabbing = grab.is_some();
                match grab {
                    Some((vertex, at, radius, strength)) => {
                        self.solver.set_grab(vertex, at, radius, strength)
                    }
                    None => self.solver.clear_grab(),
                }
                self.still = 0;
            }
            Command::Run(run) => {
                self.running = run;
                if run {
                    self.still = 0;
                }
            }
        }
    }

    /// One step, and whether the cloth has now settled.
    fn step(&mut self) -> Option<Progress> {
        let scene = self.scene.as_ref()?;
        let before = self.mesh.positions.clone();
        let moved = self.solver.step(&mut self.mesh, &scene.context());
        self.since_rebind += 1;
        let moving = self
            .mesh
            .positions
            .iter()
            .zip(&before)
            .filter(|(a, b)| a.distance(**b) > AT_REST)
            .count();
        if (moving as f32) <= STILL_MOVING * self.mesh.positions.len().max(1) as f32 {
            self.still += 1;
        } else {
            self.still = 0;
        }
        let settled =
            !self.grabbing && self.since_rebind >= MIN_STEPS && self.still >= SETTLED_STEPS;
        if settled {
            self.running = false;
        }
        Some(Progress {
            epoch: self.epoch,
            positions: self.mesh.positions.clone(),
            steps: 1,
            moved,
            stats: self.solver.stats(),
            settled,
        })
    }
}

enum Mode {
    Inline(Box<Worker>),
    Threaded {
        commands: Sender<Command>,
        progress: Receiver<Progress>,
    },
}

/// The live simulation, on its own thread or inline.
pub struct Simulation {
    mode: Mode,
    epoch: u64,
    running: bool,
}

impl Simulation {
    /// `threaded`: step on a thread of its own (the editor); otherwise step
    /// inline when asked (headless runs).
    pub fn new(solver: Box<dyn ClothSolver + Send>, threaded: bool) -> Self {
        let worker = Worker {
            solver,
            mesh: GarmentMesh::default(),
            scene: None,
            epoch: 0,
            running: false,
            grabbing: false,
            since_rebind: 0,
            still: 0,
        };
        let mode = if threaded {
            let (commands, inbox) = channel::<Command>();
            let (outbox, progress) = channel::<Progress>();
            std::thread::Builder::new()
                .name("cloth simulation".into())
                .spawn(move || run(worker, &inbox, &outbox))
                .expect("cannot start the simulation thread");
            Mode::Threaded { commands, progress }
        } else {
            Mode::Inline(Box::new(worker))
        };
        Self {
            mode,
            epoch: 0,
            running: false,
        }
    }

    fn send(&mut self, command: Command) {
        match &mut self.mode {
            Mode::Inline(worker) => worker.handle(command),
            Mode::Threaded { commands, .. } => {
                let _ = commands.send(command);
            }
        }
    }

    /// Starts over from `mesh` at rest (new topology or arrangement).
    pub fn rebind(&mut self, mesh: &GarmentMesh, scene: Scene) {
        self.epoch += 1;
        let epoch = self.epoch;
        self.send(Command::Rebind(Box::new(mesh.clone()), Box::new(scene), epoch));
    }

    pub fn refresh_materials(&mut self, scene: Scene) {
        self.send(Command::Refresh(Box::new(scene)));
    }

    pub fn set_scene(&mut self, scene: Scene) {
        self.send(Command::Scene(Box::new(scene)));
    }

    /// The cloth was moved from outside: continue from `mesh`, at rest.
    pub fn set_positions(&mut self, mesh: &GarmentMesh) {
        self.epoch += 1;
        let epoch = self.epoch;
        self.send(Command::Positions(mesh.positions.clone(), epoch));
    }

    pub fn set_grab(&mut self, vertex: u32, at: Vec3, radius: f32, strength: f32) {
        self.send(Command::Grab(Some((vertex, at, radius, strength))));
    }

    pub fn clear_grab(&mut self) {
        self.send(Command::Grab(None));
    }

    /// Whether this runs on its own thread.
    pub fn threaded(&self) -> bool {
        matches!(self.mode, Mode::Threaded { .. })
    }

    /// Threaded: keep stepping until told to stop or settled.
    pub fn run(&mut self, run: bool) {
        if run != self.running {
            self.running = run;
            self.send(Command::Run(run));
        }
    }

    /// Inline: `steps` steps now. Threaded: nothing (it steps by itself).
    /// Then everything done since the last call, newest positions only.
    pub fn poll(&mut self, steps: u32) -> Option<Progress> {
        let epoch = self.epoch;
        let mut latest: Option<Progress> = None;
        let merge = |progress: Progress, latest: &mut Option<Progress>| {
            if progress.epoch != epoch {
                return; // from before a rebind or an outside move
            }
            let (steps, moved, settled) = latest
                .as_ref()
                .map_or((0, false, false), |l| (l.steps, l.moved, l.settled));
            *latest = Some(Progress {
                steps: steps + progress.steps,
                moved: moved || progress.moved,
                settled: settled || progress.settled,
                ..progress
            });
        };
        match &mut self.mode {
            Mode::Inline(worker) => {
                for _ in 0..steps {
                    let Some(progress) = worker.step() else { break };
                    let settled = progress.settled;
                    merge(progress, &mut latest);
                    if settled {
                        break;
                    }
                }
            }
            Mode::Threaded { progress, .. } => loop {
                match progress.try_recv() {
                    Ok(p) => merge(p, &mut latest),
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                }
            },
        }
        if latest.as_ref().is_some_and(|p| p.settled) {
            self.running = false;
        }
        latest
    }
}

/// The simulation thread: follows commands, steps while running.
fn run(mut worker: Worker, inbox: &Receiver<Command>, outbox: &Sender<Progress>) {
    loop {
        // Waiting: sleep until told something.
        if !worker.running {
            match inbox.recv() {
                Ok(command) => worker.handle(command),
                Err(_) => return,
            }
        }
        loop {
            match inbox.try_recv() {
                Ok(command) => worker.handle(command),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if worker.running
            && let Some(progress) = worker.step()
            && outbox.send(progress).is_err()
        {
            return;
        }
    }
}
