import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
import imageio
from stable_baselines3 import PPO

# Ensure the root directory is in PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.environment import env

# Load configuration file
config_path = os.path.join(ROOT_DIR, 'config.json')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Extract configuration details
num_good_guys = config["num_good_guys"]
num_bad_guys = config["num_bad_guys"]

# Initialize the environment
frame_width, frame_height = 800, 600
total_agents = num_good_guys + num_bad_guys
raw_env = env(num_agents=total_agents, max_cycles=500, render_mode='rgb_array')
raw_env.reset()

# Define custom objects to replace incompatible parameters
custom_objects = {
    "clip_range": lambda _: 0.2,
    "lr_schedule": lambda _: 0.001
}

# Load models
circle_model_path = os.path.join(ROOT_DIR, 'models', 'circle_9.zip')
rectangle_model_path = os.path.join(ROOT_DIR, 'models', 'rectangle_9.zip')

if not os.path.exists(circle_model_path) or not os.path.exists(rectangle_model_path):
    raise FileNotFoundError("Required model files are missing in the 'models' directory.")

model_good = PPO.load(circle_model_path, custom_objects=custom_objects)
model_bad = PPO.load(rectangle_model_path, custom_objects=custom_objects)

# Function to create polygons
def create_polygon(image_size, num_sides, radius, center, color):
    """
    Create a fixed polygon in a 2D space.
    :param image_size: Tuple (width, height) of the image.
    :param num_sides: Number of sides of the polygon.
    :param radius: Radius of the polygon.
    :param center: Tuple (x, y) for the center of the polygon.
    :param color: Fill color of the polygon in RGBA format.
    :return: PIL Image with the polygon drawn.
    """
    image = Image.new("RGBA", image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    points = [
        (
            center[0] + radius * np.cos(2 * np.pi * i / num_sides),
            center[1] + radius * np.sin(2 * np.pi * i / num_sides),
        )
        for i in range(num_sides)
    ]

    draw.polygon(points, fill=color)
    return image

# Initialize agent positions randomly within the frame
agent_positions = {
    f"agent_{i}": (np.random.randint(50, frame_width - 50), np.random.randint(50, frame_height - 50))
    for i in range(total_agents)
}

# Create a list to store frames
frames = []

# Define fixed polygon and mountain details
polygons = [
    {"num_sides": 5, "radius": 60, "center": (150, 350), "color": (255, 165, 0, 255)},  # Orange pentagon
    {"num_sides": 6, "radius": 60, "center": (400, 400), "color": (0, 128, 0, 255)},  # Green hexagon
    {"num_sides": 3, "radius": 70, "center": (600, 150), "color": (0, 0, 255, 255)}   # Blue triangle
]

# Define visual properties
agent_radius = 7
good_color = (0, 0, 255, 255)  # Blue fill
bad_color = (255, 0, 0, 255)  # Red fill

# Run the simulation
step = 0
for agent in raw_env.agent_iter():
    observation, reward, termination, truncation, info = raw_env.last()

    # Extract numeric part of the agent identifier
    agent_id = int(agent.split("_")[1])

    if agent_id < num_good_guys:
        # Good guys (blue agents)
        action = model_good.predict(observation, deterministic=True)[0]
        x, y = agent_positions[f"agent_{agent_id}"]
        agent_positions[f"agent_{agent_id}"] = (
            np.clip(x + np.random.randint(-10, 10), 0, frame_width),
            np.clip(y + np.random.randint(-10, 10), 0, frame_height)
        )
    else:
        # Bad guys (red agents)
        action = model_bad.predict(observation, deterministic=True)[0]
        x, y = agent_positions[f"agent_{agent_id}"]
        agent_positions[f"agent_{agent_id}"] = (
            np.clip(x + np.random.randint(-10, 10), 0, frame_width),
            np.clip(y + np.random.randint(-10, 10), 0, frame_height)
        )

    if termination or truncation:
        action = None

    # Step the environment
    raw_env.step(action)

    # Capture frames periodically
    if step % 10 == 0:
        frame = raw_env.render()
        frame_image = Image.fromarray(frame).resize((frame_width, frame_height))
        draw = ImageDraw.Draw(frame_image)

        # Draw agents
        for agent_name, pos in agent_positions.items():
            x, y = pos
            color = good_color if int(agent_name.split("_")[1]) < num_good_guys else bad_color
            draw.ellipse([(x - agent_radius, y - agent_radius), (x + agent_radius, y + agent_radius)], fill=color)

        # Draw polygons
        for polygon in polygons:
            polygon_image = create_polygon(
                image_size=(frame_width, frame_height),
                num_sides=polygon["num_sides"],
                radius=polygon["radius"],
                center=polygon["center"],
                color=polygon["color"]
            )
            frame_image = Image.alpha_composite(frame_image.convert("RGBA"), polygon_image)

        frames.append(frame_image)

    step += 1

# Save the frames as a GIF
output_path = os.path.join(ROOT_DIR, 'outputs', 'simulation.gif')
if frames:
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Simulation saved as GIF at: {output_path}")
else:
    print("No frames were captured during the simulation.")
