# script used in data recollection

from moviepy.editor import VideoFileClip

def extract_audio_from_video(video_file, output_audio_file):
    # Load the video file
    video = VideoFileClip(video_file)
    
    # Extract the audio
    audio = video.audio
    
    # Write the audio to a file
    audio.write_audiofile(output_audio_file)

# Example usage
video_file = "input.mp4"  # Path to your input video file
output_audio_file = "output.mp3"  # Path to save the extracted audio file

extract_audio_from_video(video_file, output_audio_file)
