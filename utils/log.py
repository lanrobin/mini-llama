import logging
import os
import inspect

class Logger:
    def __init__(self, log_dir="~/llama_logs", log_name="llama.log", level=logging.INFO):
        """
        Initialize the Logger.
        
        Args:
            log_dir (str): Directory to save log files.
            log_name (str): Name of the log file.
            level (int): Logging level.
        """
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(level)

        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Prevent adding handlers multiple times if the logger is retrieved again
        if not self.logger.handlers:
            file_path = os.path.join(log_dir, log_name)
            
            # Create handlers
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            stream_handler = logging.StreamHandler()
            
            # Create formatter
            # Format: [Time] [Level] [File:Line] [Class.Method] - Message
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(filename_custom)s:%(lineno_custom)d] [%(classname)s.%(method_name)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    def _get_extra_info(self):
        """
        Get stack information to find the caller's filename, class, method, and line number.
        """
        stack = inspect.stack()
        # stack[0] is _get_extra_info
        # stack[1] is info/debug/error methods in this class
        # stack[2] is the caller
        try:
            frame = stack[2]
            filename = os.path.basename(frame.filename)
            lineno = frame.lineno
            method_name = frame.function
            
            class_name = "Global"
            # Try to find 'self' or 'cls' to get class name
            if 'self' in frame.frame.f_locals:
                class_name = frame.frame.f_locals['self'].__class__.__name__
            elif 'cls' in frame.frame.f_locals:
                class_name = frame.frame.f_locals['cls'].__name__
                
            return {
                'filename_custom': filename,
                'lineno_custom': lineno,
                'classname': class_name,
                'method_name': method_name
            }
        except Exception:
            return {
                'filename_custom': "unknown",
                'lineno_custom': 0,
                'classname': "unknown",
                'method_name': "unknown"
            }
        finally:
            del stack

    def info(self, message):
        self.logger.info(message, extra=self._get_extra_info())

    def debug(self, message):
        self.logger.debug(message, extra=self._get_extra_info())

    def warning(self, message):
        self.logger.warning(message, extra=self._get_extra_info())

    def error(self, message):
        self.logger.error(message, extra=self._get_extra_info())

    def critical(self, message):
        self.logger.critical(message, extra=self._get_extra_info())

if __name__ == "__main__":
    # Test execution
    log = Logger()
    log.info("Test Info Message")
    
    class TestClass:
        def test_method(self):
            log.info("Message from TestClass")
            
    tc = TestClass()
    tc.test_method()
