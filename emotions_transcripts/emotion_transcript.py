import cv2
import speech_recognition as sr
import datetime
from fer import FER

class SnDBot:
    def __init__(self, output_file="transcript.txt"):
        self.cap = None
        self.recognizer = sr.Recognizer()
        self.output_file = output_file
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_detector = FER(mtcnn=True)

    def start_capture(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise Exception("error: doesn't open the camera or video file")

    def detect_speaker_and_emotion(self, frame):
        emotions = self.emotion_detector.detect_emotions(frame)
        
        if emotions:
            face = max(emotions, key=lambda x: max(x['emotions'].values()))
            bbox = face['box']
            emotions_dict = face['emotions']
            
            dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])
            
            return (bbox[0], bbox[1], bbox[2], bbox[3]), dominant_emotion
        return None, None

    def transcribe_audio(self, audio_data):
        try:
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return "[recognition error]"

    def document(self, text, emotion, timestamp):
        with open(self.output_file, 'a', encoding='utf-8') as f:
            emotion_text = f" [Emotion: {emotion[0]} ({emotion[1]:.2f})]" if emotion else "."
            f.write(f"[{timestamp}]{emotion_text} {text}\n")

    def process_frame(self, frame):
        speaker_box, emotion = self.detect_speaker_and_emotion(frame)
        
        if speaker_box is not None:
            x, y, w, h = speaker_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if emotion:
                emotion_text = f"{emotion[0]} ({emotion[1]:.2f})"
                cv2.putText(frame, emotion_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                bar_length = int(emotion[1] * 100)
                cv2.rectangle(frame, (x, y-30), (x+bar_length, y-20), (0, 255, 0), -1)
        
        return frame, emotion

    def run(self):
        if self.cap is None:
            self.start_capture()

        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                processed_frame, current_emotion = self.process_frame(frame)
                cv2.imshow('Speaker Analysis', processed_frame)
                
                try:
                    audio = self.recognizer.listen(source, timeout=0.1, phrase_time_limit=5)
                    text = self.transcribe_audio(audio)
                    if text:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.document(text, current_emotion, timestamp)
                except sr.WaitTimeoutError:
                    pass

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    bot = SnDBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nstopping speaker analysis bot")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if bot.cap is not None:
            bot.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
