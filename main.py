# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import whisper
import ssl


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def make_transcribe(filename, model_type="base"):
    model = whisper.load_model(model_type)
    # result = model.transcribe(filename, task="translate", language="polish")
    result = model.transcribe(filename, language="polish")
    return result["text"]


def detect_language(filename, model_type="base"):
    model = whisper.load_model(model_type)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    return whisper.decode(model, mel, options)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ssl._create_default_https_context = ssl._create_unverified_context
    with open('data/out.txt', 'w') as f:
        f.write(make_transcribe("data/test.ts"))
        f.close()
    print('it is done ..')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
