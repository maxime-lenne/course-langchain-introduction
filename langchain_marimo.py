import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        ![LangChain](img/langchain.jpeg)

        # Introduction à LangChain

        **LangChain** est un framework open source conçu pour construire des applications
        d'intelligence artificielle autour des modèles de langage (LLMs) comme GPT, Claude ou Mistral.
        Il a la capacité de se connecter aux LLMs, à des sources de données, des outils,
        des chaînes de raisonnement et des moyens de stockage pour créer des systèmes interactifs et dynamiques.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ![LangChain components](img/langchain_components.png)

        ## Composants principaux

        - **LLM** : Les moteurs de raisonnement et génération de texte
        - **Prompts** : Outils pour construire des prompts dynamiques et réutilisables
        - **Chains** : Séquences logiques d'appels pour créer des pipelines IA
        - **Memory** : Gestion de la mémoire conversationnelle
        - **Agents** : Choix dynamique d'actions avec des outils disponibles
        - **Documents Loader, Text Splitters, Indexes, Vector DB** : Chaîne d'ingestion de connaissances
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        # 1. Chargement du modèle LLM local

        Dans cette section, nous chargeons un modèle de langage local grâce à **Ollama**.
        Cela permet de travailler avec un **LLM directement sur notre machine**,
        sans connexion à une API externe.
        """
    )
    return


@app.cell
def __():
    import os
    from dotenv import load_dotenv
    from langchain_ollama import ChatOllama

    # Chargement des clés d'API depuis .env
    load_dotenv(override=True)

    # Chargement du modèle local
    model = ChatOllama(model="llama3")
    return load_dotenv, model, os, ChatOllama


@app.cell
def __(mo):
    mo.md(
        """
        # 2. Requête basique

        Nous pouvons envoyer une première requête simple au modèle via la méthode `.invoke()`.
        """
    )
    return


@app.cell
def __(model, mo):
    # Requête simple au modèle
    result_basic = model.invoke(
        "Résous ce problème de mathématiques. Quel est le résultat de la division de 4 par 2 ?"
    )
    mo.md(result_basic.content)
    return (result_basic,)


@app.cell
def __(mo):
    mo.md(
        """
        ## Exercice 1

        Utilisez le modèle pour transformer une phrase simple en la réécrivant dans un style littéraire spécifique.

        1. Envoyez une requête directe via `.invoke()` contenant une instruction claire, une phrase source et le style souhaité
        2. Affichez le résultat
        """
    )
    return


@app.cell
def __(model, mo):
    # Votre code ici
    exercice_1_result = None
    mo.md("Complétez l'exercice ci-dessus")
    return (exercice_1_result,)


@app.cell
def __(mo):
    mo.md(
        """
        # 3. Conversations avec le modèle

        Plutôt que de tout écrire dans une seule phrase, il est recommandé de distinguer différents types de messages :

        - `SystemMessage` : définit le rôle ou le comportement attendu du modèle
        - `HumanMessage` : correspond à ce que vous demandez au modèle
        - `AIMessage` : représente une réponse précédente du modèle
        """
    )
    return


@app.cell
def __():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    return AIMessage, HumanMessage, SystemMessage


@app.cell
def __(mo):
    mo.md("### 3.1 Conversation sans mémoire (stateless)")
    return


@app.cell
def __(model, mo, SystemMessage, HumanMessage):
    # Messages structurés pour guider le modèle
    messages_stateless = [
        SystemMessage(content="Résous ce problème de mathématiques"),
        HumanMessage(content="Quel est le résultat de la division de 4 par 2 ?")
    ]

    result_stateless = model.invoke(messages_stateless)
    mo.md(result_stateless.content)
    return messages_stateless, result_stateless


@app.cell
def __(mo):
    mo.md("### 3.2 Conversation avec mémoire (stateful)")
    return


@app.cell
def __(model, mo, SystemMessage, HumanMessage, AIMessage):
    # Conversation simulée en plusieurs étapes
    messages_stateful = [
        SystemMessage(content="Résous ce problème de mathématiques"),
        HumanMessage(content="Quel est le résultat de la division de 4 par 2 ?"),
        AIMessage(content="Le résultat de la division de 4 par 2 est égal à 2."),
        HumanMessage(content="Et 8 multiplié par 4 ?"),
    ]

    result_stateful = model.invoke(messages_stateful)
    mo.md(result_stateful.content)
    return messages_stateful, result_stateful


@app.cell
def __(mo):
    mo.md(
        """
        ## Exercice 2

        Créez une liste `messages` avec :
        - un `SystemMessage` indiquant que l'IA est un expert dans un domaine de votre choix
        - un `HumanMessage` qui pose une question à l'IA

        Envoyez cette liste à `model.invoke()` et affichez la réponse.
        """
    )
    return


@app.cell
def __(mo):
    # Votre code ici
    mo.md("Complétez l'exercice ci-dessus")
    return


@app.cell
def __(mo):
    mo.md(
        """
        # 4. Conversations avec Prompt Templates

        `ChatPromptTemplate` permet de structurer proprement les messages envoyés au modèle
        en distinguant les rôles (system, human, assistant).
        """
    )
    return


@app.cell
def __():
    from langchain.prompts import ChatPromptTemplate
    return (ChatPromptTemplate,)


@app.cell
def __(mo):
    mo.md("### 4.1 Prompt à rôle unique (human)")
    return


@app.cell
def __(model, mo, ChatPromptTemplate):
    # Template simple "tout en un"
    template_simple = "Tu es un expert en mathématiques. Calcule le double de {value_1}, puis celui de {value_2}"
    chat_prompt_simple = ChatPromptTemplate.from_template(template_simple)

    prompt_value_simple = chat_prompt_simple.invoke({"value_1": 12, "value_2": 34})
    result_simple = model.invoke(prompt_value_simple)

    mo.md(result_simple.content)
    return template_simple, chat_prompt_simple, prompt_value_simple, result_simple


@app.cell
def __(mo):
    mo.md("### 4.2 Prompt à rôles multiples")
    return


@app.cell
def __(model, mo, ChatPromptTemplate):
    # Template structuré avec rôles explicites
    chat_prompt_multi = ChatPromptTemplate.from_messages([
        ("system", "Tu es un expert en mathématiques et un pédagogue dans ce domaine."),
        ("human", "Calcule le double de {value_1}, puis celui de {value_2}")
    ])

    prompt_value_multi = chat_prompt_multi.invoke({"value_1": 12, "value_2": 34})
    result_multi = model.invoke(prompt_value_multi)

    mo.md(result_multi.content)
    return chat_prompt_multi, prompt_value_multi, result_multi


@app.cell
def __(mo):
    mo.md(
        """
        ## Exercice 3

        Construisez un assistant capable d'adopter le style d'un philosophe célèbre.

        1. Créez un `ChatPromptTemplate` avec un message system définissant l'IA comme un philosophe `{philosopher}`
           et un message human contenant une question `{question}`
        2. Testez avec différents philosophes (Socrate, Nietzsche, etc.)
        """
    )
    return


@app.cell
def __(mo):
    # Votre code ici
    mo.md("Complétez l'exercice ci-dessus")
    return


if __name__ == "__main__":
    app.run()
