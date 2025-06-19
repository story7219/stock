# -*- coding: utf-8 -*-
"""
음성 합성 모듈 (VoiceSynthesizer)
- gTTS를 사용하여 텍스트를 음성(MP3)으로 변환합니다.
- playsound를 사용하여 생성된 음성 파일을 재생합니다.
- 음성 파일은 임시로 생성되고 재생 후 삭제됩니다.
"""
import os
import logging
from gtts import gTTS
from playsound import playsound
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

class VoiceSynthesizer:
    """
    텍스트를 음성으로 변환하고 재생하는 클래스.
    """
    def __init__(self, lang='ko'):
        """
        VoiceSynthesizer를 초기화합니다.
        :param lang: 변환할 언어 (기본값: 'ko' - 한국어)
        """
        self.lang = lang
        self.temp_dir = "temp_audio"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        logger.info(f"🔊 음성 합성기 초기화 완료 (언어: {self.lang})")

    def speak(self, text: str, save_to_file: bool = False) -> Optional[str]:
        """
        주어진 텍스트를 음성으로 변환합니다.
        - save_to_file=True: 음성 파일을 저장하고 경로를 반환합니다. (웹 앱용)
        - save_to_file=False: 음성을 즉시 재생하고 파일을 삭제합니다. (기본값)
        
        :param text: 음성으로 변환할 텍스트
        :param save_to_file: 파일을 저장하고 경로를 반환할지 여부
        :return: save_to_file=True일 경우 파일 경로, 아닐 경우 None
        """
        if not text:
            logger.warning("음성으로 변환할 텍스트가 없습니다.")
            return None
            
        temp_filename = os.path.join(self.temp_dir, f"speech_{uuid.uuid4()}.mp3")
        
        try:
            # 1. 텍스트를 음성으로 변환
            tts = gTTS(text=text, lang=self.lang)
            
            # 2. 임시 파일로 저장
            tts.save(temp_filename)
            logger.info(f"음성 파일 생성: {temp_filename}")
            
            if save_to_file:
                # 파일 경로를 반환하고, 파일은 삭제하지 않음
                return temp_filename
            else:
                # 즉시 재생
                print("🔊 AI 분석 결과를 음성으로 재생합니다...")
                playsound(temp_filename)
                return None # 재생 후 경로 반환 필요 없음
            
        except Exception as e:
            logger.error(f"❌ 음성 변환 또는 재생 중 오류 발생: {e}", exc_info=True)
            print("오류: 음성을 재생할 수 없습니다. 시스템 설정을 확인해주세요.")
            # 오류 발생 시 생성된 파일이 있다면 삭제
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as remove_e:
                    logger.error(f"❌ 오류 발생 후 임시 파일 삭제 실패: {remove_e}")
            return None
            
        finally:
            # 즉시 재생 모드일 때만 파일 삭제
            if not save_to_file and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                    logger.info(f"임시 음성 파일 삭제: {temp_filename}")
                except Exception as e:
                    logger.error(f"❌ 임시 음성 파일 삭제 실패: {e}")

    def cleanup(self):
        """
        임시 오디오 디렉토리에 남아있는 모든 파일을 정리합니다.
        """
        if os.path.exists(self.temp_dir):
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {file_path}, 오류: {e}")
            logger.info("🔊 임시 오디오 파일 정리 완료.")


if __name__ == '__main__':
    # 모듈 직접 실행 시 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    synthesizer = VoiceSynthesizer()
    
    # 1. 즉시 재생 테스트
    print("\n--- 즉시 재생 테스트 ---")
    test_text_1 = "안녕하세요? FlashStockAI 음성 즉시 재생 기능 테스트입니다."
    synthesizer.speak(test_text_1)
    
    # 2. 파일 저장 테스트
    print("\n--- 파일 저장 테스트 ---")
    test_text_2 = "이것은 파일로 저장하기 위한 테스트 음성입니다."
    saved_path = synthesizer.speak(test_text_2, save_to_file=True)
    if saved_path:
        print(f"음성 파일이 성공적으로 저장되었습니다: {saved_path}")
        # 웹 앱에서는 이 경로를 st.audio에 사용합니다.
        # 테스트 후 파일 삭제
        os.remove(saved_path)
    else:
        print("음성 파일 저장에 실패했습니다.")

    synthesizer.cleanup() 