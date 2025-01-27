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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein Assistent, der dabei hilft, gendergerechte Sätze in ausführlichere, inklusive Formulierungen zu "
                    "übersetzen. Die Umwandlung soll explizit alle Geschlechteridentitäten ansprechen und den Satz in einer länger "
                    "gefassten Form präsentieren. Hier sind einige Beispiele, wie du den Satz umformulieren könntest: "
                )
            },
            {
                "role": "user",
                "content": (
                    "Beispiel 1: 'Die Student:innen sind sehr fleißig' -> 'Die Studenten, Studentinnen und Personen anderer "
                    "geschlechtlicher Identität, die an einer Hochschule studieren, sind sehr fleißig'."
                    "Beispiel 2: 'Influencer*innen haben ganz viele Abonnent*innen' -> 'Influencerinnen, Influencer und Personen mit anderen geschlechtlichen Identitäten, "
                    "die in den sozialen Medien aktiv sind, haben eine große Anzahl an Abonnentinnen, Abonnenten und weiteren Personen, die ihre Inhalte verfolgen, unabhängig "
                    "davon, wie diese sich in Bezug auf Geschlecht oder Geschlechtsidentität definieren.'"
                    "Beispiel 3: 'Das Missymagazin sucht einen Abonnierenden' -> 'Das Missymagazin sucht eine Person, die das Magazin abonniert, unabhängig von deren Geschlechtsidentität, "
                    "sei es eine Frau, ein Mann oder eine Person mit einer anderen geschlechtlichen Identität.'"
                    f"Bitte wandle folgenden gendergerechten Satz in eine detaillierte, inklusive Form um: {text}"
                )
            }
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    example_sentence = "Die Student:innen sind sehr fleißig."

    print("Original: ", example_sentence)
    enhanced_data = find_inclusive_form(example_sentence)
    print("Longer form: ", enhanced_data)