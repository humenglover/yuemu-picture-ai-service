"""
文本审核服务 - DFA 纯净版
审核流程：
  仅使用高性能 DFA 算法匹配敏感词，包含忽略干扰字符功能。
  取代原有极其消耗内存的 TextCNN ONNX 模型。
"""
import os
import time
import re

from utils.log_utils import app_logger

# ========================= DFA 算法实现 =========================

class DFAFilter:
    """
    确定有穷自动机 (DFA) 过滤器
    支持海量词库毫秒级匹配，并支持忽略干扰符（如空格、*、- 等）
    """
    def __init__(self):
        self.keyword_chains = {}  # Trie 树根节点
        self.delimit = '\x00'    # 终止符
        # 干扰项忽略列表：非字母数字及非中文字符
        self.skip_chars = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9]")

    def add(self, keyword: str):
        if not keyword:
            return
        keyword = keyword.lower().strip()
        chars = keyword
        if not chars:
            return
        level = self.keyword_chains
        for i in range(len(chars)):
            if chars[i] in level:
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):
                    break
                for j in range(i, len(chars)):
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0}
                break
        if i == len(chars) - 1:
            level[self.delimit] = 0

    def parse(self, path: str):
        """兼容 textfilter 的 parse()，读取词库文件"""
        if not os.path.exists(path):
            app_logger.warning(f"词库文件不存在: {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            for keyword in f:
                self.add(keyword.strip())

    def check(self, message: str) -> list[str]:
        """
        检查文本中是否包含敏感词
        返回: 命中的敏感词列表
        """
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            found_word = ""
            
            for i in range(start, len(message)):
                char = message[i]
                
                # 跳过干扰字符
                if self.skip_chars.match(char):
                    step_ins += 1
                    continue
                
                if char in level:
                    step_ins += 1
                    found_word += char
                    if self.delimit in level[char]:
                        # 命中一个敏感词
                        ret.append(message[start:start + step_ins])
                        # 这里退出循环，不再匹配更长的词，以命中首个即告命中
                        break
                    level = level[char]
                else:
                    break
            start += 1
        return list(set(ret))

    def filter(self, message: str, repl="*") -> str:
        """替换敏感词"""
        message = message.lower()
        ret = list(message)
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for i in range(start, len(message)):
                char = message[i]
                if self.skip_chars.match(char):
                    step_ins += 1
                    continue
                if char in level:
                    step_ins += 1
                    if self.delimit in level[char]:
                        for j in range(start, start + step_ins):
                            ret[j] = repl
                        start = start + step_ins - 1
                        break
                    level = level[char]
                else:
                    break
            start += 1
        return "".join(ret)


# ========================= 路径配置 =========================
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "moderation")
_WORD_LIST_PATH = os.path.join(_BASE_DIR, "word_list.txt")
_ILLEGAL_PATH = os.path.join(_BASE_DIR, "illegal.txt")
_SUSPECTED_PATH = os.path.join(_BASE_DIR, "suspected_illegal.txt")


# ========================= 服务类 =========================

class CommentModerationService:
    """
    文本审核服务（单例）
    升级为纯 DFA 高性能匹配，移除原先依赖 jieba 与 onnxruntime 的 TextCNN 推理。
    """

    def __init__(self):
        self._ready = False
        self._dfa = DFAFilter()

    def init(self):
        """初始化所有数据资源"""
        app_logger.info("初始化文本审核服务 (纯 DFA 版本)...")

        # 将原有三个词库合并注入同一个 DFAFilter
        app_logger.info(f"  加载词库文件: {_ILLEGAL_PATH}")
        self._dfa.parse(_ILLEGAL_PATH)

        app_logger.info(f"  加载词库文件: {_SUSPECTED_PATH}")
        self._dfa.parse(_SUSPECTED_PATH)

        app_logger.info(f"  加载词库文件: {_WORD_LIST_PATH}")
        self._dfa.parse(_WORD_LIST_PATH)

        self._ready = True
        app_logger.info(f"文本审核服务初始化成功 | 内存开销已降至极低。")

    # ==================== 公开接口 ====================

    def moderate(self, comments: list[str], mode: str = "accurate") -> dict:
        """
        执行文本审核
        返回格式（与原接口兼容）:
          {
            "results": { "<评论文本>": { "label": 0|1, "score": 0.xx, "index": 0 } },
            "costSeconds": 0.001
          }
        """
        if not self._ready:
            raise RuntimeError("CommentModerationService 尚未初始化，请先调用 init()")

        start = time.time()
        results = {}

        for i, comment in enumerate(comments):
            hits = self._dfa.check(comment)
            if hits:
                filtered_comment = self._dfa.filter(comment, "*")
                app_logger.warning(f"  [DFA] 命中敏感词: {hits} | 原文: {comment} | 替换后: {filtered_comment}")
                results[comment] = {'label': 1, 'score': 1.0, 'index': i, 'filtered_text': filtered_comment}
            else:
                results[comment] = {'label': 0, 'score': 0.0, 'index': i, 'filtered_text': comment}

        intercepted_count = sum(1 for r in results.values() if r['label'] == 1)
        app_logger.info(
            f"  [DFA] 审核完毕: 命中 {intercepted_count}/{len(comments)} 条"
        )

        elapsed = round(time.time() - start, 6)
        return {'results': results, 'costSeconds': elapsed}


# ========================= 全局单例 =========================
comment_moderation_service = CommentModerationService()
