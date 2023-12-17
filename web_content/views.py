from .models import Post
import os
from django.utils import timezone
from django.shortcuts import render
from PIL import Image
import openai
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import textstat
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.stem import PorterStemmer
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

file_path = os.path.abspath("static/")
nlp = spacy.load('en_core_web_sm')
personal_stop_words = ['hagrid', 'ron', 'snape', 'hermione', 'petunia', 'vernon',
                       'nicholas', 'flamel', 'dobby', 'whoa', 'harry', 'nt']
stop_words = stopwords.words('english')
for text in personal_stop_words:
    stop_words.append(text)
df_data_dictionary = pd.read_csv(f'{file_path}/Datasets/Data_Dictionary.csv', encoding='latin1')
df_dialogue = pd.read_csv(f'{file_path}/Datasets/Dialogue.csv', encoding='latin1')
df_chapters = pd.read_csv(f'{file_path}/Datasets/Chapters.csv', encoding='latin1')
df_characters = pd.read_csv(f'{file_path}/Datasets/Characters.csv', encoding='latin1')
df_movies = pd.read_csv(f'{file_path}/Datasets/Movies.csv', encoding='latin1')
df_places = pd.read_csv(f'{file_path}/Datasets/Places.csv', encoding='latin1')
# Сбрасываем ограничения на количество выводимых рядов
pd.set_option('display.max_rows', None)
# Сбрасываем ограничения на число столбцов
pd.set_option('display.max_columns', None)
# Сбрасываем ограничения на количество символов в записи
pd.set_option('display.max_colwidth', None)
# df_dialogue[df_dialogue['Chapter ID'] <35 ]
prompt_for_gpt = ''
dialog_history = []
gpt_history = []
gpt_history.append(f"Type an answer, using this additional information about the text: \n")

# -----------------------------------------------------------------
def no_stop_words(text):
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text


# Функція чистки тексту (усі зайві символи)
def clean(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
    text = text.replace("  ", " ")
    return text


# Лематизація
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text


# Частини мови
def get_pos_frequency(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    pos_frequency = {}
    for token in doc:
        pos = token.pos_
        if pos in pos_frequency:
            pos_frequency[pos] += 1
        else:
            pos_frequency[pos] = 1
    return pos_frequency


# Тональність
def get_sentiment(text):
    blob = TextBlob(text)
    # Знаходження тональності
    sentiment = blob.sentiment
    return sentiment


# Функція для визначення фраз у тексті
def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)


def extract_phrases_spacy(text):
    doc = nlp(text)
    phrases = [sent.text for sent in doc.sents]
    return phrases


# Функція для пошуку найпоширеніших слів та фраз
def find_most_common_words(text, count):
    # Розбиваємо текст на слова
    text = re.sub('\.+', ' ', text)
    words = word_tokenize(text.lower())
    # Знаходимо найпоширеніші слова
    common_words = FreqDist(words).most_common(count)
    return common_words


def get_most_common_phrases(text, n, top_k):
    # Токенізація тексту на слова
    words = text.split()
    # Створення n-грам фраз
    phrases = [' '.join(ngram) for ngram in ngrams(words, n)]
    # Знаходження найбільш поширених фраз
    common_phrases = Counter(phrases).most_common(top_k)
    return common_phrases


# Створення wordcloud для візуалізації частотності слів
def generate_wordcloud(text):
    meta_mask = np.array(Image.open(f'{file_path}/meta.jpg'))
    font_path = f'{file_path}/fonts/Germgoth.ttf'
    wc = WordCloud(background_color='white',
                   mask=meta_mask,
                   contour_width=2,
                   contour_color='white',
                   font_path=font_path,
                   colormap='tab10',
                   width=1200, height=800).generate(text)
    wc.to_file(f'web_content/static/wordcloud_image.png')
    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')


# -----------------------------------------------------------------
def analyze(name_of_character):
    gpt_history = []
    character = df_characters.merge(df_dialogue, left_on='Character ID',
                                    right_on='Character ID',
                                    suffixes=('_left', '_right'))
    character = character[
        character['Character Name'] == name_of_character]  # Harry Potter , Ron Weasley , Hermione Granger
    character_name = df_characters[df_characters['Character Name'] == name_of_character]
    character_name = character_name['Character Name']
    character_name = ''.join(character_name)
    Harry = character
    Harry = Harry.drop(['Species', 'Character Name',
                        'Gender', 'House', 'Patronus',
                        'Wand (Wood)', 'Wand (Core)', 'Dialogue ID',
                        'Chapter ID', 'Place ID'],
                       axis='columns')
    # -----------------------------------------------------------------
    if not Harry['Dialogue'].empty:
        # Заміна екранованих апострофів
        original_text = np.array(Harry['Dialogue'].str.replace(r"\\'", " ' "))
        original_text_string = ' '.join(original_text)
        cleaned_text = [clean_text(text) for text in original_text]
        # Об'єднання рядків
        string = ' '.join(cleaned_text)
        # print(string)
    else:
         print("Діалоги для Harry Potter відсутні.")
    # -----------------------------------------------------------------
    sent_tokens = []
    cleaned_text_1 = []
    cleaned_text = []  # Текст, очищений від пунктуації та особливих закінчень
    tokens = []  # Список для докенізації
    filtered_text_list = []
    filtered_sentences = []
    stopwords_cleaned = []  # Список для тексту, очищений від стоп-слів
    for text in Harry['Dialogue']:
        text = clean_text(text)
        cleaned_text.append(text)
        text = word_tokenize(text)
        tokens.append(text)
        text = [word for word in text if word not in stop_words]
        stopwords_cleaned.append(text)
    Harry['Cleaned_Dialogue'] = cleaned_text
    Harry['Tokens_Dialogue'] = tokens
    Harry['noStopwords_Dialogue'] = stopwords_cleaned
    # -----------------------------------------------------------------
    for text in Harry['Dialogue']:
        text = sent_tokenize(text)
        sent_tokens += [' '.join(sublist) for sublist in text if sublist and any(sublist)]
    for text in sent_tokens:
        text = clean_text(text)
        cleaned_text_1.append(text)
    filtered_sentences = [' '.join(filter(lambda word: word not in stop_words, sentence.split())) for sentence in
                          cleaned_text_1]
    filtered_text_list = list(filter(bool, filtered_sentences))
    filtered_string = ''
    filtered_string = ' '.join(filtered_text_list)
    # -----------------------------------------------------------------
    # Завантаження моделі мови для англійської
    nlp = spacy.load("en_core_web_sm")
    # Текст для обробки
    text = original_text_string
    # Обробка тексту SpaCy
    doc = nlp(text)
    # 1. Пошук вузлів-прийменників або сполучників з залежними вузлами
    subclauses = {chunk.text for chunk in doc.noun_chunks if chunk.root.dep_ == "relcl"}
    # 2. Пошук вузлів особових займенників, що є коренем піддерева
    subordinate_clauses = {sent.text for sent in doc.sents if any(token.dep_ == "nsubj" and
                                                                  token.head.dep_ == "ROOT"
                                                                  for token in sent)}
    # 3. Пошук вузлів для сурядних речень
    # Створення множини складних речень у тексті
    complex_sentences = {sent.text for sent in doc.sents if len(list(sent.root.children)) > 1}
    # Порахувати загальну кількість речень
    total_sentences = len(list(doc.sents))
    # Вивести результати
    #print(f"\nЗагальна кількість речень: {total_sentences}")
    #print(f"Кількість складних речень: {len(subclauses.union(subordinate_clauses, complex_sentences))}")
    #print(
    #    f"Відсоток складних речень від загальної кількості: {len(subclauses.union(subordinate_clauses, complex_sentences)) / total_sentences * 100}%")
    average_sentence_length = textstat.avg_sentence_length(text)
    #print(f"Середня довжина речення: {average_sentence_length} слів")
    word_count = textstat.lexicon_count(text)
    #print(f'Кількість слів в тексті: {word_count}')
    average_word_length = textstat.avg_letter_per_word(text)
    #print(f"Середня довжина слова: {average_word_length} символів")

    # -------------------------------------------
    string = clean(string)
    lemmatized_text = lemmatize_text(string)
    #print(lemmatized_text)
    # -------------------------------------------
    nested_sentences = []
    for sentence in Harry['Dialogue']:
        nested_sentences.append([sentence])
    #print(nested_sentences)
    # -------------------------------------------
    prompt_for_gpt = f'\nYour must answer as a {character_name} character. \n'
    string2 = clean_text(lemmatized_text)
    string2 = no_stop_words(string2)
    generate_wordcloud(string2)
    most_common_words = find_most_common_words(string2, 10)
    pos_frequency = get_pos_frequency(original_text_string)
    #print("Частотність частин мови:", pos_frequency)
    sentiment = get_sentiment(original_text_string)
    #print("----------------------------------------------")
    #print("Тональність тексту: ", sentiment)
    #print("----------------------------------------------")
    #print("Найпоширеніші слова: ")
    prompt_for_gpt += (
        f"The percentage of complex sentences from the total number: {len(subclauses.union(subordinate_clauses, complex_sentences)) / total_sentences * 100}% \n Average sentence length: {average_sentence_length} words \n Number of words in the text: {word_count} \n Average word length: {average_word_length} symbols \n Part of speech frequency: {pos_frequency} \n Sentiment of the text: {sentiment} \n ")
    prompt_for_gpt += (f"Most frequent words of {character_name}: \n ")
    for word, count in most_common_words:
        prompt_for_gpt += (f"{word}: {count} \n ")
        print(f"{word}: {count}")
    #print("----------------------------------------------")
    # ---------------------------------------------
    full_text = ' '.join([sentence[0] for sentence in nested_sentences])
    full_text = clean_text(full_text)
    # full_text = no_stop_words(full_text)
    for n in range(2, 6):
        prompt_for_gpt += (f"Most common {n}-gram phrases of {character_name}: \n ")
        #print(f"Найпоширеніші фрази з {n} грамами: ")
        # get_most_common_phrases
        most_common_phrases = get_most_common_phrases(full_text, n, top_k=5)
        for phrase, count in most_common_phrases:
            #print(f"{phrase}: {count} разів")
            prompt_for_gpt += (f"{phrase}: {count} \n ")
        #print("-----------------")
    prompt_for_gpt += (f"Use most frequent words to create an answer on my question. \n ")
    return prompt_for_gpt


def wordcloud_view(request):
    text = "Текст для генерації WordCloud"  # Замініть це на ваш текст або отримайте його з моделі
    generate_wordcloud(text)  # Викликайте вашу функцію для генерації WordCloud
    return render(request, 'web_content/harry.html')


# Create your views here.
def web_content_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'web_content/web_content_list.html', {'posts': posts})


prompt = f"{prompt_for_gpt} \n "


def harry(request):
    prompt_for_gpt = ''
    openai.api_key = "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"
    ending = True
    if request.method == 'POST':
        print('lalala')
        user_input = request.POST['user_input']
        dialog_history.append(f"{user_input}")
        gpt_history.append(user_input)
        # Формування запиту до OpenAI на основі історії діалогу
        prompt = "\n".join(gpt_history)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        # Отримання та виведення відповіді від GPT
        bot_reply = response.choices[0].text.strip()
        dialog_history.append(f"{bot_reply}")

        if user_input.lower() == "close":
            dialog_history.clear()
            ending = False
    else:
        gpt_history.clear()
        dialog_history.clear()
        prompt_for_gpt = analyze('Harry Potter')
        # gpt_history.append(f"Type an answer, using this additional information about the text: \n")
        gpt_history.append(prompt_for_gpt)
        print(prompt_for_gpt)

    return render(request, 'web_content/harry.html',
                  {'textarea': prompt_for_gpt, 'dialog_history': dialog_history, 'ending': ending})


def hermione(request):
    prompt_for_gpt = ''
    openai.api_key = "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"
    ending = True
    if request.method == 'POST':
        print('lalala')
        user_input = request.POST['user_input']
        dialog_history.append(f"{user_input}")
        gpt_history.append(user_input)
        # Формування запиту до OpenAI на основі історії діалогу
        prompt = "\n".join(gpt_history)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.9,
        )
        # Отримання та виведення відповіді від GPT
        bot_reply = response.choices[0].text.strip()
        dialog_history.append(f"{bot_reply}")

        if user_input.lower() == "close":
            dialog_history.clear()
            ending = False
    else:
        gpt_history.clear()
        dialog_history.clear()
        prompt_for_gpt = analyze('Hermione Granger')
        # gpt_history.append(f"Type an answer, using this additional information about the text: \n")
        gpt_history.append(prompt_for_gpt)
        print(prompt_for_gpt)

    return render(request, 'web_content/hermione.html',
                  {'textarea': prompt_for_gpt, 'dialog_history': dialog_history, 'ending': ending})


def ron(request):
    prompt_for_gpt = ''
    openai.api_key = "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"
    ending = True
    if request.method == 'POST':
        user_input = request.POST['user_input']
        dialog_history.append(f"{user_input}")
        gpt_history.append(user_input)
        # Формування запиту до OpenAI на основі історії діалогу
        prompt = "\n".join(gpt_history)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.9,
        )
        # Отримання та виведення відповіді від GPT
        bot_reply = response.choices[0].text.strip()
        dialog_history.append(f"{bot_reply}")

        if user_input.lower() == "close":
            dialog_history.clear()
            ending = False
    else:
        gpt_history.clear()
        dialog_history.clear()
        prompt_for_gpt = analyze('Ron Weasley')
        #gpt_history.append(f"Type an answer, using this additional information about the text: \n")
        gpt_history.append(prompt_for_gpt)
        print(prompt_for_gpt)

    return render(request, 'web_content/ron.html',
                  {'textarea': prompt_for_gpt, 'dialog_history': dialog_history, 'ending': ending})