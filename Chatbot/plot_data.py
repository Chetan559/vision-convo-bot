import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_observations(all_words, tags, xy):
    # Word Frequency Plot
    word_counts = Counter([word for words, _ in xy for word in words])
    common_words, frequencies = zip(*word_counts.most_common(10))
    
    plt.figure(figsize=(12, 5))
    plt.bar(common_words, frequencies, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Frequent Words')
    plt.xticks(rotation=45)
    plt.show()

    # Tag Distribution Plot
    tag_counts = Counter([tag for _, tag in xy])
    labels, values = zip(*tag_counts.items())

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    plt.title('Tag Distribution')
    plt.show()
