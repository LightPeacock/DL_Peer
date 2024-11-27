import torch
import pyttsx3
import tkinter as tk
from tkinter import messagebox
from diffusers import DiffusionPipeline
from gtts import gTTS
import os
import subprocess
import numpy as np
from PIL import Image

# Load the DiffusionPipeline model for 3D generation
pipe = DiffusionPipeline.from_pretrained("Intel/ldm3d-4c")

# Save the texture (RGB image) as a .png file
def save_texture(rgb_image, texture_filename="texture.png"):
    rgb_image.save(texture_filename)
    return texture_filename

# Function to save mesh as .obj with texture
def save_mesh_with_texture_as_obj(vertices, faces, uv_coords, filename, texture_file):
    with open(filename, 'w') as f:
        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write UV coordinates
        for uv in uv_coords:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        # Write faces with texture coordinates
        for face in faces:
            f.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")

    # Create material file for texture
    with open(filename.replace('.obj', '.mtl'), 'w') as mtl_file:
        mtl_file.write(f"newmtl material_0\n")
        mtl_file.write(f"map_Kd {texture_file}\n")  

    return filename

# Function to generate vertices, faces, and UV mapping from depth and RGB images
def depth_map_to_mesh_with_uv(depth_map, width, height):
    depth_values = np.array(depth_map)

    vertices = []
    faces = []
    uv_coords = []

    # Create vertices and UV coordinates using depth map and image dimensions
    for y in range(height):
        for x in range(width):
            z = depth_values[y, x] / 255.0  # Normalize depth value
            vertices.append([x, y, z])
            uv_coords.append([x / width, y / height])  # UV mapping is normalized by image size

    # Create faces (triangles) between adjacent vertices
    for y in range(height - 1):
        for x in range(width - 1):
            idx = y * width + x
            faces.append([idx + 1, idx + 2, idx + width + 1])  # First triangle
            faces.append([idx + 2, idx + width + 2, idx + width + 1])  # Second triangle

    return vertices, faces, uv_coords

# Function to generate the 3D model using DiffusionPipeline and save with texture
def generate_3d_model(prompt):
    # Use the diffusion pipeline to generate the output
    output = pipe(prompt)
    
    # Get the depth and RGB images from the output
    depth_image = output.depth[0]
    rgb_image = output.rgb[0]

    # Save RGB image as a texture
    texture_file = save_texture(rgb_image)

    # Get the image size for mesh generation
    width, height = depth_image.size

    # Convert depth map to 3D mesh with UV mapping
    vertices, faces, uv_coords = depth_map_to_mesh_with_uv(depth_image, width, height)
    
    # Save the mesh as a .obj file with texture
    generated_model_path = save_mesh_with_texture_as_obj(vertices, faces, uv_coords, "generated_model.obj", texture_file)
    
    return generated_model_path

# Text-to-speech functionality
def text_to_speech(prompt):
    engine = pyttsx3.init()
    engine.say(prompt)
    engine.runAndWait()

# Display the 3D model using a viewer like Blender (or a default image viewer for images)
def display_3d_model(model_path):
    if os.name == 'posix':  # For macOS/Linux
        subprocess.run(['open', model_path])
    elif os.name == 'nt':  # For Windows
        os.startfile(model_path)
    else:
        subprocess.run(['xdg-open', model_path])

# GUI for the application
class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Based 3D Model Generator with Speech")
        
        self.prompt_label = tk.Label(root, text="Enter your prompt:")
        self.prompt_label.pack()
        self.prompt_entry = tk.Entry(root, width=50)
        self.prompt_entry.pack()

        self.generate_button = tk.Button(root, text="Generate 3D Model", command=self.generate_model)
        self.generate_button.pack(pady=10)

        self.speech_button = tk.Button(root, text="Speak Prompt", command=self.speak_prompt)
        self.speech_button.pack(pady=10)
    
        self.display_button = tk.Button(root, text="Display 3D Model", command=self.display_model)
        self.display_button.pack(pady=10)

        self.model_path = None

    # Function to generate the 3D model
    def generate_model(self):
        prompt = self.prompt_entry.get()  # Get the text prompt from the entry widget
        if prompt:
            self.model_path = generate_3d_model(prompt)  # Pass the prompt to the function
            messagebox.showinfo("Success", "3D model generated and saved as 'generated_model.obj'!")
        else:
            messagebox.showwarning("Input Error", "Please enter a valid prompt.")

    # Function to speak the entered prompt
    def speak_prompt(self):
        prompt = self.prompt_entry.get()
        if prompt:
            text_to_speech(prompt) 
        else:
            messagebox.showwarning("Input Error", "Please enter a valid prompt.")

    # Function to display the generated 3D model
    def display_model(self):
        if self.model_path:
            display_3d_model(self.model_path)
        else:
            messagebox.showwarning("Model Error", "Please generate a 3D model first.")

# Main function to run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()
