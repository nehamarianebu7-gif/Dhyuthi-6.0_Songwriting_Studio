!pip install crewai google-generativeai python-dotenv ipywidgets

from google.colab import output
output.enable_custom_widget_manager()

import google.generativeai as genai
from crewai import Agent, Task, Crew
import os
from crewai import LLM

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key="AIzaSyBNLS1mcGzI7YzCp4a2vuv6svJAm4bUmGg",  # Or set GOOGLE_API_KEY/GEMINI_API_KEY
    temperature=0.7
)

def gemini_tool(prompt):
    response = gemini.generate_content(prompt)
    return response.text

import ipywidgets as widgets
#theme
theme = widgets.Text(
    description="Theme:",
    placeholder="e.g. heartbreak, nostalgia"
)
#genre selection
genre = widgets.Dropdown(
    options=["Pop", "Rock", "R&B", "EDM", "Indie", "Rap","Melody"],
    description="Genre:"
)
#emotion intensity
emotion = widgets.IntSlider(
    min=0, max=10,
    description="Emotion:"
)

display(theme, genre, emotion)

user_prompt = f"""
Write original lyrics about {theme.value} in a {genre.value} style.
The emotional intensity should be {emotion.value} out of 10.
Include:
- Verse
- Chorus
- Bridge
Use vivid imagery and strong metaphors.
"""

#lyricist agent
lyricist = Agent(
    role="Lyricist",
    goal="Write compelling, vivid, singable lyrics.",
    backstory="You are an award-winning songwriter known for poetic language.",
    llm=llm,
    verbose=True
)

#emotion agent
emotion_analyst = Agent(
    role="EmotionalArcAnalyst",
    goal="Analyze emotional pacing, imagery, catharsis.",
    backstory="Expert in music psychology & affective flow.",
    llm=llm,
    verbose=True
)

#refinement agent
refiner = Agent(
    role="Refiner",
    goal="Punch up weak lines, improve rhyme & rhythm.",
    backstory="Lyric editor with 10+ years experience",
    llm=llm,
    verbose=True
)

#music theory agent
composer = Agent(
    role="Composer",
    goal="Suggest chord progressions, BPM, tempo direction.",
    backstory="Award-nominated music arranger & composer.",
    llm=llm,
    verbose=True
)

#critic agent
critic = Agent(
    role="Critic",
    goal="Score creativity, hook-strength, radio-fit.",
    backstory="Harsh but fair music critique journalist.",
    llm=llm,
    verbose=True
)

t1 = Task(
    description=f"Generate song lyrics: {user_prompt}",
    agent=lyricist,
    expected_output="Write the song lyrics with Verse,Chorus and Bridge using the given lyrics,theme and genre."
)

t2 = Task(
    description="Analyze the emotional arc and imagery of the above lyrics.",
    agent=emotion_analyst,
    expected_output="Write the lyrics with the given emotion and emotional intensity.Don't explain.",
    context=[t1]
)

t3 = Task(
    description="Refine the lyrics improving rhythm, rhyme, imagery, and originality.",
    agent=refiner,
    expected_output="Refine the lyrics improving rhythm,rhyme,imagery and originality.",
    context=[t2]
)

t4 = Task(
    description="Suggest chord progressions and BPM suitable for the refined lyrics.",
    agent=composer,
    expected_output="Suggest chord progressions and BPM suitable for the refined lyrics after the lyrics.",
    context=[t3]
)

t5 = Task(
    description="Critique the final lyrics and score them (0-10) based on catchiness in 3-4 sentences.",
    agent=critic,
    expected_output="Critique the final lyrics and score them(0-10) based on catchiness and lyrics.",
    context=[t4]
)

crew = Crew(
    agents=[lyricist, emotion_analyst, refiner, composer, critic],
    tasks=[t1, t2, t3, t4, t5],
    verbose=True
)

result = crew.kickoff()
print(result)

