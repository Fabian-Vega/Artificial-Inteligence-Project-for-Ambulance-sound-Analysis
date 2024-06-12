import os
import random
import shutil
import tkinter as tk
from tkinter import messagebox
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa

# Paths
unheard_folder = '../data/unclassified/Archi'
sirens_folder = '../data/unclassified/Archi/sirens'
ambient_folder = '../data/unclassified/Archi/ambient'
unusable_folder = '../data/unclassified/Archi/unusable'

# Function to play audio
def play_audio(file_path):
    sound = AudioSegment.from_file(file_path)
    play(sound)

# Function to move file to the specified folder
def move_file(file_path, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    shutil.move(file_path, target_folder)

# Function to load and play a random unheard audio file
def load_random_audio():
    unheard_files = [f for f in os.listdir(unheard_folder) if f.endswith('.mp3')]
    if not unheard_files:
        messagebox.showinfo("Info", "No more unheard audio files!")
        return None
    random_file = random.choice(unheard_files)
    return os.path.join(unheard_folder, random_file)

# Callback functions for buttons
def classify_as_siren():
    global current_file
    if current_file:
        move_file(current_file, sirens_folder)
        next_audio()

def classify_as_ambient():
    global current_file
    if current_file:
        move_file(current_file, ambient_folder)
        next_audio()

def classify_as_unusable():
    global current_file
    if current_file:
        move_file(current_file, unusable_folder)
        next_audio()

def next_audio():
    global current_file
    current_file = load_random_audio()
    if current_file:
        play_audio(current_file)

# Setting up the UI
root = tk.Tk()
root.title("Audio Classifier")

current_file = None

btn_siren = tk.Button(root, text="Siren", command=classify_as_siren, width=25, height=5)
btn_siren.pack(pady=10)

btn_ambient = tk.Button(root, text="Ambient Noise", command=classify_as_ambient, width=25, height=5)
btn_ambient.pack(pady=10)

btn_unusable = tk.Button(root, text="Unusable", command=classify_as_unusable, width=25, height=5)
btn_unusable.pack(pady=10)

next_audio()

root.mainloop()
