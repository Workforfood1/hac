"""
Backend для обработки результатов: XML, шифрование, отправка по почте.
ОТДЕЛЬНЫЙ модуль - не включается в основную обработку видео.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
import base64


class XMLExporter:
    """Экспорт результатов в XML формат."""
    
    @staticmethod
    def export(results: List[Dict[str, Any]], output_path: str):
        """
        Экспортирует результаты в XML.
        
        Args:
            results: список результатов обработки видео
            output_path: путь для сохранения XML
        """
        root = ET.Element('video_analysis')
        root.set('total_frames', str(len(results)))
        
        total_detections = sum(r['count'] for r in results)
        root.set('total_detections', str(total_detections))
        
        for frame_result in results:
            frame_elem = ET.SubElement(root, 'frame')
            frame_elem.set('number', str(frame_result['frame_num']))
            frame_elem.set('detections', str(frame_result['count']))
            
            for detection in frame_result['detections']:
                det_elem = ET.SubElement(frame_elem, 'detection')
                
                det_elem.set('class', str(detection.get('class_name', 'unknown')))
                det_elem.set('confidence', f"{detection.get('confidence', 0):.3f}")
                
                # Bounding box
                bbox = ET.SubElement(det_elem, 'bbox')
                bbox.set('x1', f"{detection.get('x1', 0):.1f}")
                bbox.set('y1', f"{detection.get('y1', 0):.1f}")
                bbox.set('x2', f"{detection.get('x2', 0):.1f}")
                bbox.set('y2', f"{detection.get('y2', 0):.1f}")
                
                # OCR результат
                if detection.get('ocr_text'):
                    ocr = ET.SubElement(det_elem, 'ocr')
                    ocr.set('text', detection['ocr_text'])
                    ocr.set('confidence', f"{detection.get('ocr_confidence', 0):.3f}")
        
        # Сохраняем с красивым форматированием
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        print(f"💾 XML сохранен: {output_path}")


class Encryptor:
    """Простое шифрование результатов (base64 для примера)."""
    
    @staticmethod
    def encrypt_base64(data: str) -> str:
        """Кодирует данные в base64."""
        return base64.b64encode(data.encode()).decode()
    
    @staticmethod
    def decrypt_base64(encrypted: str) -> str:
        """Декодирует данные из base64."""
        return base64.b64decode(encrypted.encode()).decode()
    
    @staticmethod
    def encrypt_file(input_path: str, output_path: str):
        """Шифрует файл результатов."""
        with open(input_path, 'r') as f:
            data = f.read()
        
        encrypted = Encryptor.encrypt_base64(data)
        
        with open(output_path, 'w') as f:
            f.write(encrypted)
        
        print(f"🔐 Файл зашифрован: {output_path}")


class EmailSender:
    """Отправка результатов по электронной почте."""
    
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
    
    def send_results(self, sender_email: str, sender_password: str,
                     recipient_email: str, results_file: str,
                     subject: str = "Video Analysis Results"):
        """
        Отправляет результаты по почте.
        
        Args:
            sender_email: email отправителя
            sender_password: пароль/токен отправителя
            recipient_email: email получателя
            results_file: путь к файлу результатов
            subject: тема письма
        """
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email.mime.multipart import MIMEMultipart
        from email import encoders
        
        try:
            # Создаем письмо
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # Текст письма
            body = """
            Привет,
            
            К этому письму приложены результаты анализа видео.
            
            С уважением,
            Video Analysis System
            """
            msg.attach(MIMEText(body, 'plain'))
            
            # Прикрепляем файл
            if Path(results_file).exists():
                with open(results_file, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename= {Path(results_file).name}')
                msg.attach(part)
            
            # Отправляем письмо
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            print(f"✉️  Письмо отправлено: {recipient_email}")
            return True
        
        except Exception as e:
            print(f"❌ Ошибка при отправке email: {e}")
            return False


class ResultsProcessor:
    """Главный класс для обработки результатов (XML, шифрование, почта)."""
    
    def __init__(self):
        self.xml_exporter = XMLExporter()
        self.encryptor = Encryptor()
        self.email_sender = EmailSender()
    
    def process_results(self, results_file: str, export_format: str = 'xml',
                       encrypt: bool = False, send_email: bool = False,
                       email_config: dict = None):
        """
        Обрабатывает результаты: экспорт, шифрование, отправка.
        
        Args:
            results_file: путь к JSON файлу результатов
            export_format: 'xml', 'json' или 'both'
            encrypt: шифровать ли результаты
            send_email: отправлять ли по почте
            email_config: конфиг для отправки email
        """
        # Загружаем результаты
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        output_files = []
        
        # 1. Экспорт в XML
        if export_format in ['xml', 'both']:
            xml_file = results_file.replace('.json', '.xml')
            XMLExporter.export(results, xml_file)
            output_files.append(xml_file)
        
        # 2. Шифрование
        if encrypt:
            if export_format in ['json', 'both']:
                encrypted_file = results_file.replace('.json', '.encrypted')
                self.encryptor.encrypt_file(results_file, encrypted_file)
                output_files.append(encrypted_file)
        
        # 3. Отправка email
        if send_email and email_config:
            recipient = email_config.get('recipient_email')
            sender = email_config.get('sender_email')
            password = email_config.get('sender_password')
            
            if all([recipient, sender, password]):
                for file_to_send in output_files:
                    self.email_sender.send_results(
                        sender_email=sender,
                        sender_password=password,
                        recipient_email=recipient,
                        results_file=file_to_send
                    )
        
        return output_files


def example_backend_usage():
    """Пример использования backend функций."""
    print("📧 Backend для XML, шифрования и почты")
    print("-" * 60)
    
    # Пример результатов
    sample_results = [
        {
            'frame_num': 1,
            'count': 2,
            'detections': [
                {
                    'x1': 100, 'y1': 200, 'x2': 300, 'y2': 400,
                    'confidence': 0.95,
                    'class': 0,
                    'class_name': 'person',
                    'ocr_text': '12345',
                    'ocr_confidence': 0.87
                }
            ]
        }
    ]
    
    # 1. Экспорт в XML
    print("\n1️⃣  Экспорт в XML:")
    XMLExporter.export(sample_results, 'results_example.xml')
    print("   ✅ XML создан")
    
    # 2. Шифрование
    print("\n2️⃣  Шифрование:")
    json_data = json.dumps(sample_results, indent=2)
    encrypted = Encryptor.encrypt_base64(json_data)
    print(f"   📝 Оригинал: {len(json_data)} символов")
    print(f"   🔐 Зашифровано: {len(encrypted)} символов")
    decrypted = Encryptor.decrypt_base64(encrypted)
    print(f"   ✅ Расшифровано: {len(decrypted)} символов")
    
    # 3. Email (требует конфигурации)
    print("\n3️⃣  Отправка email:")
    print("   ℹ️  Для отправки требуются учетные данные SMTP")
    print("   Используйте ResultsProcessor.process_results() с email_config")


if __name__ == '__main__':
    example_backend_usage()
