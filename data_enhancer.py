import openai
import os

openai_api_key = os.getenv("OPENAI_API_KEY")

def find_inclusive_form(text: str) -> str:
    """
    Creating longer text forms of gendered sentences.
    :param text: The gendered sentence to enhance.
    :return: The long form sentence.
    """
    client = openai.OpenAI(
        api_key=openai_api_key,
    )

    messages=[
        {
            "role": "system",
            "content": (
                "Du bist ein Assistent, der dabei hilft, gendergerechte Sätze in ausführlichere, inklusive Formulierungen zu "
                "übersetzen. Deine Aufgabe ist es, die Sätze so umzuformulieren, dass sie alle Geschlechteridentitäten ansprechen, "
                "wobei die Sprache detailliert und respektvoll ist. Die Umwandlungen sollten klare Beispiele und die Vielfalt "
                "von Geschlechtsidentitäten berücksichtigen. Achte darauf, dass du die Formulierungen so lange wie nötig ausführst, "
                "um die Inklusivität in allen Aspekten zu verdeutlichen."
            )
        },
        {
            "role": "user",
            "content": (
                "Beispiel 1: 'Die Student:innen sind sehr fleißig' -> 'Die Studenten, Studentinnen und alle anderen Personen, "
                "die in einer höheren Bildungseinrichtung studieren und sich mit verschiedenen Geschlechtsidentitäten identifizieren, "
                "sind sehr fleißig.'"
                "\nBeispiel 2: 'Influencer*innen haben viele Abonnent*innen' -> 'Influencerinnen, Influencer, und alle weiteren "
                "Personen, die in den sozialen Medien aktiv sind und deren Geschlechtsidentität möglicherweise nicht den traditionellen "
                "Kategorien entspricht, haben viele Abonnentinnen, Abonnenten und weitere Follower.'"
                "\nBeispiel 3: 'Das Missymagazin sucht einen Abonnierenden' -> 'Das Missymagazin sucht eine Person, die das Magazin "
                "abonniert hat, unabhängig davon, ob diese Person sich als Frau, Mann oder mit einer anderen Geschlechtsidentität "
                "identifiziert.'"
                "\nBeispiel 4: 'Der Geschäftsführer ist sehr erfahren' -> 'Die Person, die die Leitung des Unternehmens innehat, "
                "unabhängig davon, ob diese Person sich als Mann, Frau oder mit einer anderen Geschlechtsidentität identifiziert, "
                "ist sehr erfahren.'"
                "\nBeispiel 5: 'Der Leser ist aufgerufen, seine Meinung zu äußern' -> 'Die Person, die diesen Text liest, wird "
                "eingeladen, ihre Meinung zu äußern, ganz gleich, wie sie sich in Bezug auf Geschlecht und Geschlechtsidentität "
                "versteht.'"
                f"\nBitte wandle folgenden gendergerechten Satz in eine detaillierte, inklusive Form um: {text}"
            )
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    example_sentence = "Die Student:innen sind sehr fleißig."

    print("Original: ", example_sentence)
    enhanced_data = find_inclusive_form(example_sentence)
    print("Longer form: ", enhanced_data)