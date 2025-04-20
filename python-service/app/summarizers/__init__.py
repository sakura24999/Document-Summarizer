from .base_summarizer import BaseSummarizer
from .claude_summarizer import ClaudeSummarizer

__all__ = ['BaseSummarizer', 'ClaudeSummarizer']

def get_summarizer(summarizer_type='claude'):
    """
    指定されたタイプのサマライザーを返します

    Args:
        summarizer_type (str): サマライザーのタイプ

    Returns:
        BaseSummarizer: サマライザーのインスタンス
    """
    if summarizer_type.lower() == 'claude':
        return ClaudeSummarizer()
    else:
        raise ValueError(f"サポートされていないサマライザータイプ: {summarizer_type}")
