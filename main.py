# Initial Design Intent (High Level Overview):
#
# This project is an experimental neural network training framework for learning reusable
# low level control skills in a 2D physics based environment. Creatures are modeled as thrust
# and rotation driven ships operating under impulse physics, with perception provided by
# structured sensors such as goal direction and distance, obstacle proximity, and threat
# awareness. Training is designed to run at very high speed using many parallel simulated
# worlds at once, allowing large numbers of candidate policies to be evaluated simultaneously.
# Rather than training a single monolithic policy, the system is built around a mixture of
# experts approach: multiple specialist neural networks are trained independently for narrow
# skills such as navigation, obstacle avoidance, stabilization, or evasion, then frozen and
# composed via a learned gating network that blends their outputs using soft weights to produce
# smooth, fuzzy behavior transitions. The final creature brain consists of shared physics and
# action interfaces, a library of trained expert policies, and one or more gating networks that
# control behavioral mixtures programmatically. The code below represents an evolving
# implementation of this plan and may diverge from, partially implement, or experiment beyond
# this original design as development progresses.


import math
import pygame
import torch
from abc import ABC, abstractmethod


# ===============================
# Global Simulation Configuration
# ===============================

# Frames per second target
FPS = 60

# Fixed timestep for physics updates
DT = 1.0 / FPS

# Window dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# ===============================
# Global Simulation State
# ===============================

running = True
world = None   # declaration
renderer = None  # declaration

# ===============================
# World Configuration
# ===============================

# World dimensions (toroidal space)
WORLD_WIDTH = 0
WORLD_HEIGHT = 0


# ===============================
# Creature Physics Configuration
# ===============================

# Limit the creature's maximum forward thrust
MAX_THRUST = 300.0

# Limit the creature's rotational torque
MAX_TORQUE = 6.0

# Maximum linear speed (pixels per second)
MAX_SPEED = 500.0

# Linear damping to prevent infinite drifting (space friction)
LINEAR_DAMPING = 0.985

# Angular damping to prevent endless spinning
ANGULAR_DAMPING = 0.99


# ===============================
# Controller Interface
# ===============================

# This is the abstract base class for all creature controllers.
# A controller's job is to observe the creature (and later the world)
# and decide what control inputs to apply.
#
# This mirrors a Java-style interface using Python's ABC system.
class Controller(ABC):

    @abstractmethod
    def get_control(self, creature):
        """
        Return control intent for the creature.

        Expected return:
            thrust (float): forward/backward thrust in range [-1, 1]
            turn   (float): rotational input in range [-1, 1]
        """
        pass

# ===============================
# Keyboard Controller
# ===============================

# This controller maps WASD keyboard input to creature controls.
# It is a temporary stand-in for a neural network controller.
class KeyboardController(Controller):

    def get_control(self, creature):
        # Read current keyboard state
        keys = pygame.key.get_pressed()

        # Initialize control outputs
        thrust = 0.0
        turn = 0.0

        # Forward / backward thrust
        if keys[pygame.K_w]:
            thrust += 1.0
        if keys[pygame.K_s]:
            thrust -= 1.0

        # Rotational control
        if keys[pygame.K_a]:
            turn -= 1.0
        if keys[pygame.K_d]:
            turn += 1.0

        # Clamp values to expected control range
        thrust = max(-1.0, min(1.0, thrust))
        turn = max(-1.0, min(1.0, turn))

        return thrust, turn
    

# ===============================
# World Class
# ===============================

# The World represents an independent simulation environment.
#
# Motivation:
# - We want to run many worlds simultaneously in the future
# - Each world may have different sizes, entities, and rules
# - Worlds must be updatable without rendering (headless training)
#
# Design principle:
# - The World owns spatial rules (bounds, wrapping)
# - The World owns collections of entities
# - The World does NOT know about pygame or input
class World:
    def __init__(self, width, height):
        # World dimensions (in world units, currently pixels)
        self.width = width
        self.height = height

        # Entities contained in this world
        self.creatures = []

        # Control flags
        self.paused = False     # If True, world does not advance
        self.visible = True    # If False, world is not rendered

    def add_creature(self, creature):
        # Register a creature with this world.
        #
        # Motivation:
        # - Worlds may contain many creatures later
        # - Allows centralized update and reset logic
        self.creatures.append(creature)

    def update(self):
        # Advance the simulation by one timestep.
        #
        # Motivation:
        # - This becomes the core "step()" function used during training
        # - Allows worlds to be paused, stepped independently, or batched
        if self.paused:
            return

        for creature in self.creatures:
            creature.update()
            self.apply_wrapping(creature)

    def apply_wrapping(self, creature):
        # Apply toroidal (wrap around) boundary conditions to a creature.
        #
        # Motivation:
        # - Boundary rules are properties of the world, not the creature
        # - Different worlds may have different topology later
        # - Centralizing this logic enables multi world simulation
        pos = creature.pos

        # Horizontal wrapping
        if pos[0] < 0:
            pos[0] += self.width
        elif pos[0] >= self.width:
            pos[0] -= self.width

        # Vertical wrapping
        if pos[1] < 0:
            pos[1] += self.height
        elif pos[1] >= self.height:
            pos[1] -= self.height



# ===============================
# Creature Class
# ===============================

# This is the main game object representing a controllable creature/ship.
# It owns:
# - physical state (position, velocity, angle)
# - physics integration logic
# - a clean control interface
#
# Important design idea:
# The creature does NOT know where control comes from.
# Today it is keyboard input.
# Later it will be a neural network.
class Creature:
    def __init__(self, start_pos, controller):
        # Position in world space
        self.pos = torch.tensor(start_pos, dtype=torch.float32)

        # Linear velocity
        self.vel = torch.zeros(2)

        # Orientation and angular velocity
        self.angle = torch.tensor(0.0)
        self.ang_vel = torch.tensor(0.0)

        # Controller responsible for deciding movement
        self.controller = controller

        # ===============================
        # Geometry (Local Space)
        # ===============================

        # List of polygons that define the creature's body.
        #
        # Motivation:
        # - Creatures may be composed of multiple shapes later
        # - Shapes may be decided by a genome
        # - Renderer should not invent geometry
        #
        # Convention:
        # - Points are in local space
        # - Forward direction is +X
        self.polygons = [
            [
                ( 8,  0),
                (-8,  5),
                (-8, -5),
            ]
        ]
    

    # Update physics and control for one timestep
    def update(self):
        # Ask the controller what control inputs to apply
        thrust, turn = self.controller.get_control(self)

        # Apply control intent to physics
        self.apply_control(thrust, turn)

        # Advance the physics simulation
        self.physics_step()


    # Apply control inputs to the creature's physics state
    def apply_control(self, thrust, turn):
        # Apply player (or future NN) control input to the creature.
        #
        # thrust: [-1, 1]  -> forward/backward force
        # turn:   [-1, 1]  -> left/right rotational force

        # Convert current angle into a forward direction vector
        forward = torch.tensor([
            math.cos(self.angle),
            math.sin(self.angle)
        ])

        # Convert thrust input into linear acceleration
        acceleration = forward * (thrust * MAX_THRUST)

        # Integrate acceleration into velocity
        self.vel += acceleration * DT

        # Convert turn input into angular acceleration
        self.ang_vel += turn * MAX_TORQUE * DT


    # Advance the physics simulation by one timestep
    def physics_step(self):
        # Apply damping to simulate friction and stabilize motion
        self.vel *= LINEAR_DAMPING
        self.ang_vel *= ANGULAR_DAMPING

        # Clamp maximum linear speed
        max_speed = MAX_SPEED  # pixels per second
        speed = torch.linalg.norm(self.vel)

        if speed > max_speed:
            self.vel = self.vel / speed * max_speed

        # Integrate velocity into position
        self.pos += self.vel * DT

        # Integrate angular velocity into orientation
        self.angle += self.ang_vel * DT



# ===============================
# World Renderer
# ===============================

# The WorldRenderer is responsible for drawing a World to a surface.
#
# Motivation:
# - Allows worlds to be simulated without rendering (headless training)
# - Allows multiple worlds to be drawn differently
# - Allows overlaying multiple worlds in one window
#
# Design principle:
# - Renderer observes world state
# - Renderer does NOT modify simulation state
class WorldRenderer:
    def __init__(self, surface):
        # Surface to draw onto (pygame surface)
        self.surface = surface

    def draw(self, world):
        # Clear background
        self.surface.fill((20, 20, 20))

        # Draw all creatures in the world
        for creature in world.creatures:
            self.draw_creature(creature)

    def draw_creature(self, creature):
        # Draw all polygons that make up the creature.
        #
        # Motivation:
        # - Renderer does not assume creature shape
        # - Supports multi polygon bodies
        # - Supports genome defined geometry later

        angle = creature.angle
        pos = creature.pos

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        for polygon in creature.polygons:
            transformed = []

            for x, y in polygon:
                # Rotate from local space into world space
                rx = x * cos_a - y * sin_a
                ry = x * sin_a + y * cos_a

                # Translate into world position
                sx = int(pos[0] + rx)
                sy = int(pos[1] + ry)

                transformed.append((sx, sy))

            pygame.draw.polygon(self.surface, (240, 240, 240), transformed)




# ===============================
# Input Handling
# ===============================

# Read keyboard state and convert it into a control vector.
# This mirrors what a neural network will output later.
def get_keyboard_control():
    keys = pygame.key.get_pressed()

    thrust = 0.0
    turn = 0.0

    # Forward / backward thrust
    if keys[pygame.K_w]:
        thrust += 1.0
    if keys[pygame.K_s]:
        thrust -= 1.0

    # Rotational control
    if keys[pygame.K_a]:
        turn += 1.0
    if keys[pygame.K_d]:
        turn -= 1.0

    # Clamp inputs to expected range
    thrust = max(-1.0, min(1.0, thrust))
    turn = max(-1.0, min(1.0, turn))

    return thrust, turn

def checkEvents():
    global running, screen, WORLD_WIDTH, WORLD_HEIGHT

    world = World(WORLD_WIDTH, WORLD_HEIGHT)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(
                (event.w, event.h),
                pygame.RESIZABLE
            )
            WORLD_WIDTH, WORLD_HEIGHT = screen.get_size()

            # Update world dimensions to match window
            world.width = WORLD_WIDTH
            world.height = WORLD_HEIGHT

            # Update renderer surface
            renderer.surface = screen


# ===============================
# Main Game Loop
# ===============================

# Main entry point for the simulation
def main():
    global WORLD_WIDTH, WORLD_HEIGHT, world, renderer

    world = World(WORLD_WIDTH, WORLD_HEIGHT)


    # Initialize pygame and window
    pygame.init()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.RESIZABLE
    )

    # Query actual screen size from pygame
    WORLD_WIDTH, WORLD_HEIGHT = screen.get_size()
    pygame.display.set_caption("PyTorch Physics Creature Demo")

    # Create clock for fixed timestep
    clock = pygame.time.Clock()

    # Create world renderer
    renderer = WorldRenderer(screen)

    # Create a keyboard-based controller
    controller = KeyboardController()

    # Create a single creature at the center of the screen
    creature = Creature(start_pos=(WORLD_WIDTH / 2, WORLD_HEIGHT / 2), controller=controller)


    world.add_creature(creature)

    # Main loop
    while running:
        # Maintain fixed timestep
        clock.tick(FPS)

        # Handle input events
        checkEvents()

        # Update world
        world.update()

        # Render world
        renderer.draw(world)

        # Update display
        pygame.display.flip()

        
    # Clean up and exit
    pygame.quit()


# ===============================
# Entry Point
# ===============================

# Run the main simulation loop if this file is executed directly
if __name__ == "__main__":

    # Run the main simulation
    main()
