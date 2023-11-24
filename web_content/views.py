from django.shortcuts import render
from .models import Post
from django.utils import timezone

dialog_history = []
# Create your views here.
def web_content_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'web_content/web_content_list.html', {'posts': posts})


from django.shortcuts import render
import openai


def chat_bot(request):
    openai.api_key = "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"

    ending = True

    if request.method == 'POST':
        user_input = request.POST['user_input']
        dialog_history.append(f"You: {user_input}")

        # Формування запиту до OpenAI на основі історії діалогу
        prompt = "\n".join(dialog_history)
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        # Отримання та виведення відповіді від GPT
        bot_reply = response.choices[0].text.strip()
        dialog_history.append(f"Bot: {bot_reply}")

        if user_input.lower() == "close":
            ending = False

    return render(request, 'web_content/chat_bot.html', {'dialog_history': dialog_history, 'ending': ending})
