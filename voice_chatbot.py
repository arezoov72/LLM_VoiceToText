import openai
import speech_recognition as sr
import pyttsx3
from transformers import pipeline, set_seed

# === CONFIGURATION ===
USE_OPENAI = False  # Change to True for online mode
openai.api_key = "YOUR_OPENAI_API_KEY"  # Set only if online mode



# === INIT TOOLS ===
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
local_generator = pipeline('text-generation', model='distilgpt2')
set_seed(42)

# === FUNCTIONS ===
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def list_microphones():
    mics = sr.Microphone.list_microphone_names()
    print("🎧 Available microphone devices:")
    for i, name in enumerate(mics):
        print(f"  [{i}] {name}")
    return mics

def listen(device_index=0):
    try:
        with sr.Microphone(device_index=device_index) as source:
            print(f"🎤 Using device [{device_index}]... Speak now.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text
    except AssertionError as e:
        print(f"🚫 Microphone error: {e}")
        return ""
    except AttributeError as e:
        print(f"💥 Device error: {e} — this likely means the device isn't available.")
        return ""
    except sr.UnknownValueError:
        print("⚠️ Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"❌ Speech recognition failed: {e}")
        return ""
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return ""

mics = list_microphones()

# Optionally, auto-select the first valid device
device_index = None
for i, name in enumerate(mics):
    if "microphone" in name.lower():
        device_index = i
        break

if device_index is None:
    print("⚠️ No microphone found. Using default index 0.")
    device_index = 0

def get_online_response(prompt):
    print("🌐 Getting response from OpenAI...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def get_offline_response(prompt):
    print("🖥 Generating offline response...")
    response = local_generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"].strip()

# === MAIN LOOP ===
print("🤖 Voice Chatbot Started (Type Ctrl+C or say 'exit' to quit)")
while True:
    try:
        user_input = listen()
        if not user_input:
            continue
        if "exit" in user_input.lower():
            speak("Goodbye!")
            break

        if USE_OPENAI:
            response = get_online_response(user_input)
        else:
            response = get_offline_response(user_input)

        print(f"🤖 Bot: {response}")
        speak(response)

    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        break
while True:
    user_input = listen(device_index=device_index)