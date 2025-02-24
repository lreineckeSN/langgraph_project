from datetime import datetime
from typing import Dict, List, Literal, TypedDict, Union, Any, Optional

from langchain.tools import tool
from langchain_openai import AzureOpenAI, OpenAI, AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()
# Konfiguration
LLM = AzureChatOpenAI(azure_deployment="gpt-4o-mini", )


# Schema für die Transaktionsdaten
class Transaction(TypedDict):
    id: str
    timestamp: str
    sender_account: str
    receiver_account: str
    amount: float
    currency: str
    description: str
    is_realtime: bool


# Schema für den Systemzustand
class State(TypedDict):
    transaction: Transaction
    rule_based_result: Optional[Dict[str, Any]]
    ml_model_result: Optional[Dict[str, Any]]
    coordinator_decision: Optional[Literal["accept", "reject", "manual_review"]]
    explanation: Optional[str]
    fraud_manager_questions: Optional[List[str]]
    fraud_manager_decision: Optional[Literal["accept", "reject"]]
    current_step: Literal[
        "initial", "rule_analysis", "ml_analysis", "coordination", "explanation", "fraud_manager", "final"]


# Simulierte Datenbank-Tools für den ReAct-Agenten

@tool
def search_customer_history(account_id: str) -> Dict[str, Any]:
    """
    Durchsucht die Kundenhistorie nach früheren Transaktionen und Verhaltensmustern.

    Args:
        account_id: Die Kontonummer des Kunden

    Returns:
        Ein Dictionary mit Kundeninformationen und Transaktionshistorie
    """
    # Simulierte Daten - in einer realen Implementierung würde hier eine Datenbankabfrage stehen
    customer_data = {
        "12345678": {
            "name": "Max Mustermann",
            "account_since": "2015-03-12",
            "risk_score": 0.2,  # Niedrig
            "typical_transaction_volume": 500.0,
            "typical_recipients": ["DE89370400440532013000", "DE27100777770209299700"],
            "unusual_activity_last_30_days": False,
            "recent_transactions": [
                {"date": "2023-04-01", "amount": 450.0, "recipient": "DE89370400440532013000"},
                {"date": "2023-03-25", "amount": 35.0, "recipient": "DE27100777770209299700"},
                {"date": "2023-03-20", "amount": 520.0, "recipient": "DE89370400440532013000"}
            ]
        },
        "87654321": {
            "name": "Erika Musterfrau",
            "account_since": "2018-11-05",
            "risk_score": 0.7,  # Höher
            "typical_transaction_volume": 200.0,
            "typical_recipients": ["DE11520513735120710131"],
            "unusual_activity_last_30_days": True,
            "recent_transactions": [
                {"date": "2023-04-02", "amount": 1800.0, "recipient": "DE41630400530051700432"},  # Ungewöhnlich hoch
                {"date": "2023-03-28", "amount": 150.0, "recipient": "DE11520513735120710131"},
                {"date": "2023-03-27", "amount": 170.0, "recipient": "DE11520513735120710131"}
            ]
        }
    }

    return customer_data.get(account_id, {"error": "Kunde nicht gefunden"})


def analyze_transaction_pattern(account_id: str, transaction_amount: float) -> Dict[str, Any]:
    """
    Analysiert, ob die aktuelle Transaktion vom typischen Muster des Kunden abweicht.

    Args:
        account_id: Die Kontonummer des Kunden
        transaction_amount: Der Betrag der aktuellen Transaktion

    Returns:
        Ein Dictionary mit der Analyse des Transaktionsmusters
    """
    # Simulierte Daten
    customer_data = search_customer_history(account_id)

    if "error" in customer_data:
        return {"error": "Kunde nicht gefunden"}

    typical_volume = customer_data["typical_transaction_volume"]
    deviation = abs(transaction_amount - typical_volume) / typical_volume if typical_volume > 0 else 0

    return {
        "typical_volume": typical_volume,
        "current_transaction": transaction_amount,
        "deviation_percentage": deviation * 100,
        "is_unusual": deviation > 1.0,  # Mehr als 100% Abweichung
        "risk_assessment": "hoch" if deviation > 1.0 else "mittel" if deviation > 0.5 else "niedrig"
    }


@tool
def check_recipient_risk(recipient_account: str) -> Dict[str, Any]:
    """
    Überprüft den Empfänger auf potenzielle Risiken.

    Args:
        recipient_account: Die Kontonummer des Empfängers

    Returns:
        Ein Dictionary mit Risikobewertung des Empfängers
    """
    # Simulierte Daten - Bekannte risikoreiche Konten
    high_risk_accounts = ["DE41630400530051700432", "DE88500105175964254899"]
    medium_risk_accounts = ["DE22500105176474540931", "DE36500105172861764774"]

    if recipient_account in high_risk_accounts:
        return {
            "account": recipient_account,
            "risk_level": "hoch",
            "previous_fraud_cases": 3,
            "recommendation": "Transaktion ablehnen"
        }
    elif recipient_account in medium_risk_accounts:
        return {
            "account": recipient_account,
            "risk_level": "mittel",
            "previous_fraud_cases": 1,
            "recommendation": "Manuelle Überprüfung"
        }
    else:
        return {
            "account": recipient_account,
            "risk_level": "niedrig",
            "previous_fraud_cases": 0,
            "recommendation": "Transaktion akzeptieren"
        }


# Komponenten des Systems

def rule_based_analysis(state: State):
    """Führt eine regelbasierte Analyse der Transaktion durch."""
    transaction = state["transaction"]

    # Regelbasierte Prüfungen
    risk_factors = []
    risk_score = 0.0

    # Prüfung auf hohe Beträge
    if transaction["amount"] > 1000:
        risk_factors.append("Hoher Transaktionsbetrag")
        risk_score += 0.3

    # Prüfung auf Echtzeitüberweisung
    if transaction["is_realtime"]:
        risk_factors.append("Echtzeitüberweisung")
        risk_score += 0.1

    # Prüfung des Empfängers
    recipient_check = check_recipient_risk(transaction["receiver_account"])
    if recipient_check["risk_level"] == "hoch":
        risk_factors.append("Risikoreicher Empfänger")
        risk_score += 0.5
    elif recipient_check["risk_level"] == "mittel":
        risk_factors.append("Empfänger mit mittlerem Risiko")
        risk_score += 0.2

    result = {
        "risk_score": min(risk_score, 1.0),  # Maximal 1.0
        "risk_factors": risk_factors,
        "timestamp": datetime.now().isoformat(),
        "recommendation": "reject" if risk_score > 0.7 else "manual_review" if risk_score > 0.3 else "accept"
    }

    return {"rule_based_result": result, "current_step": "ml_analysis"}


def ml_model_analysis(state: State):
    """Simuliert eine ML-Modell-Analyse der Transaktion."""
    transaction = state["transaction"]

    # Analyse des Transaktionsmusters
    pattern_analysis = analyze_transaction_pattern(
        transaction["sender_account"],
        transaction["amount"]
    )

    # Simulierte ML-Bewertung basierend auf verschiedenen Faktoren
    features = {
        "amount": transaction["amount"],
        "is_realtime": 1 if transaction["is_realtime"] else 0,
        "unusual_pattern": 1 if pattern_analysis.get("is_unusual", False) else 0,
        "time_of_day": datetime.fromisoformat(transaction["timestamp"]).hour,
        "day_of_week": datetime.fromisoformat(transaction["timestamp"]).weekday(),
    }

    # Simuliertes ML-Ergebnis
    # In einer realen Implementierung würde hier ein tatsächliches ML-Modell verwendet
    unusual_amount = pattern_analysis.get("is_unusual", False)
    weekend = features["day_of_week"] >= 5
    night_time = features["time_of_day"] < 6 or features["time_of_day"] > 22

    fraud_probability = 0.1  # Basiswahrscheinlichkeit

    if unusual_amount:
        fraud_probability += 0.4
    if weekend and features["amount"] > 1000:
        fraud_probability += 0.2
    if night_time and features["amount"] > 500:
        fraud_probability += 0.3
    if features["is_realtime"] and features["amount"] > 1000:
        fraud_probability += 0.2

    fraud_probability = min(fraud_probability, 1.0)  # Maximal 1.0

    result = {
        "fraud_probability": fraud_probability,
        "confidence": 0.85,
        "features_importance": {
            "transaction_amount": 0.35,
            "unusual_pattern": 0.25,
            "time_factors": 0.2,
            "realtime_factor": 0.2
        },
        "recommendation": "reject" if fraud_probability > 0.7 else "manual_review" if fraud_probability > 0.3 else "accept",
        "timestamp": datetime.now().isoformat()
    }

    return {"ml_model_result": result, "current_step": "coordination"}


def coordinator_decision(state: State) -> Union[State, Literal["generate_explanation", "final_accept"]]:
    """Koordiniert die Entscheidung basierend auf regelbasierter und ML-Analyse."""
    rule_result = state["rule_based_result"]
    ml_result = state["ml_model_result"]

    # Kombinierte Bewertung
    rule_rec = rule_result["recommendation"]
    ml_rec = ml_result["recommendation"]

    # Entscheidungslogik
    if rule_rec == "reject" or ml_rec == "reject":
        decision = "reject"
    elif rule_rec == "manual_review" or ml_rec == "manual_review":
        decision = "manual_review"
    else:
        decision = "accept"

    new_state = {**state, "coordinator_decision": decision, "current_step": "explanation"}

    # Entscheidung über den weiteren Verlauf
    if decision == "accept" and not state["transaction"]["is_realtime"]:
        return "final_accept"
    else:
        return "generate_explanation"


# ReAct-Agent für Erklärungen
def create_explanation_agent():
    """Erstellt einen ReAct-Agenten für die Erklärungsgenerierung."""
    # Tools für den Agenten
    tools = [
        search_customer_history,
        check_recipient_risk
    ]

    # Erstellung des ReAct-Agenten
    agent = create_react_agent(LLM, tools)

    # Anpassung des Agenten für Erklärungsgenerierung
    def explain_decision(state: State):
        transaction = state["transaction"]
        rule_result = state["rule_based_result"]
        ml_result = state["ml_model_result"]
        decision = state["coordinator_decision"]

        # Prompt für den Erklärungsagenten
        prompt = f"""
        Als erklärender KI-Agent sollst du die Entscheidung des Fraud-Detection-Systems nachvollziehbar erklären.

        Transaktion:
        - ID: {transaction['id']}
        - Sender: {transaction['sender_account']}
        - Empfänger: {transaction['receiver_account']}
        - Betrag: {transaction['amount']} {transaction['currency']}
        - Beschreibung: {transaction['description']}
        - Echtzeitüberweisung: {"Ja" if transaction['is_realtime'] else "Nein"}

        Regelbasierte Analyse:
        - Risikobewertung: {rule_result['risk_score']}
        - Risikofaktoren: {', '.join(rule_result['risk_factors']) if rule_result['risk_factors'] else "Keine"}

        ML-Modell Analyse:
        - Betrugswahrscheinlichkeit: {ml_result['fraud_probability']}
        - Konfidenz: {ml_result['confidence']}

        Entscheidung: {decision.upper()}

        Bitte nutze die dir zur Verfügung stehenden Tools, um Informationen über den Kunden und seine Transaktionshistorie zu sammeln.
        Erkläre dann in natürlicher Sprache, warum diese Entscheidung getroffen wurde.
        Berücksichtige dabei alle relevanten Faktoren und gib eine klare, strukturierte Erklärung.
        """

        # Ausführung des Agenten
        result = agent.invoke({"input": prompt})
        explanation = result["output"]

        return {"explanation": explanation, "current_step": "fraud_manager"}

    return explain_decision


# Interaktion mit dem Fraud Manager
def fraud_manager_interaction():
    """Erstellt eine Funktion für die Interaktion mit dem Fraud Manager."""
    # Tools für den Interaktionsagenten
    tools = [
        search_customer_history,
        check_recipient_risk
    ]

    # Erstellung des ReAct-Agenten
    agent = create_react_agent(LLM, tools)

    def handle_question(state: State, question: str):
        """Verarbeitet eine Frage des Fraud Managers."""
        transaction = state["transaction"]

        # Prompt für den Agenten
        prompt = f"""
        Als erklärender KI-Agent sollst du auf die Frage des Fraud Managers antworten.

        Transaktion:
        - ID: {transaction['id']}
        - Sender: {transaction['sender_account']}
        - Empfänger: {transaction['receiver_account']}
        - Betrag: {transaction['amount']} {transaction['currency']}

        Frage des Fraud Managers:
        {question}

        Bitte nutze die dir zur Verfügung stehenden Tools, um alle relevanten Informationen zu sammeln und
        eine fundierte, klare Antwort zu geben. Verweise dabei auf konkrete Daten und Fakten.
        """

        # Ausführung des Agenten
        result = agent.invoke({"input": prompt})
        answer = result["output"]

        # Aktualisieren der Frageliste (für Audit-Zwecke)
        questions = state.get("fraud_manager_questions", [])
        questions.append({"question": question, "answer": answer})

        return {"fraud_manager_questions": questions}

    return handle_question


def make_final_decision(state: State, decision: Literal["accept", "reject"]):
    """Speichert die finale Entscheidung des Fraud Managers."""
    return {"fraud_manager_decision": decision, "current_step": "final"}


# Hauptgraph
def create_fraud_detection_graph():
    """Erstellt den Hauptgraphen für das Fraud-Detection-System."""
    workflow = StateGraph(State)

    # Knoten hinzufügen
    workflow.add_node("rule_analysis", rule_based_analysis)
    workflow.add_node("ml_analysis", ml_model_analysis)
    workflow.add_node("coordination", coordinator_decision)
    workflow.add_node("create_explanation", create_explanation_agent())

    # Kanten hinzufügen
    workflow.set_entry_point("rule_analysis")
    workflow.add_edge("rule_analysis", "ml_analysis")
    workflow.add_edge("ml_analysis", "coordination")
    workflow.add_conditional_edges(
        "coordination",
        lambda state: state["coordinator_decision"],
        {
            "accept": END,
            "reject": "create_explanation",
            "manual_review": "create_explanation"
        }
    )
    workflow.add_edge("create_explanation", END)

    return workflow.compile()


# Initialisierung und Ausführung

def process_transaction(transaction: Transaction):
    """Verarbeitet eine neue Transaktion im System."""
    # Initialer Zustand
    initial_state = {
        "transaction": transaction,
        "rule_based_result": None,
        "ml_model_result": None,
        "coordinator_decision": None,
        "explanation": None,
        "fraud_manager_questions": None,
        "fraud_manager_decision": None,
        "current_step": "initial"
    }

    # Graph erstellen und ausführen
    graph = create_fraud_detection_graph()
    result = graph.invoke(initial_state)

    return result


# Beispiel für eine Transaktion
example_transaction = {
    "id": "TX12345678",
    "timestamp": "2023-04-03T15:30:25",
    "sender_account": "87654321",
    "receiver_account": "DE41630400530051700432",
    "amount": 2000.0,
    "currency": "EUR",
    "description": "Zahlung für neue Möbel",
    "is_realtime": True
}


# Interactive Fraud Manager Interface
def simulate_fraud_manager_interface(transaction_result: State):
    """Simuliert eine Schnittstelle für den Fraud Manager."""
    print("\n=== FRAUD MANAGER INTERFACE ===")
    print(f"Transaktion: {transaction_result['transaction']['id']}")
    print(f"Betrag: {transaction_result['transaction']['amount']} {transaction_result['transaction']['currency']}")
    print(f"Systemempfehlung: {transaction_result['coordinator_decision'].upper()}")
    print("\nErklärung:")
    print(transaction_result['explanation'])

    # In einer realen Anwendung würde hier eine tatsächliche Benutzeroberfläche stehen
    # mit der Möglichkeit, Fragen zu stellen und Entscheidungen zu treffen

    # Simulierte Frage des Fraud Managers
    question_handler = fraud_manager_interaction()
    updated_state = question_handler(
        transaction_result,
        "Welche ungewöhnlichen Aktivitäten gab es in der letzten Zeit auf diesem Konto?"
    )

    print("\nFrage des Fraud Managers:")
    print("Welche ungewöhnlichen Aktivitäten gab es in der letzten Zeit auf diesem Konto?")
    print("\nAntwort:")
    print(updated_state['fraud_manager_questions'][-1]['answer'])

    # Simulierte Entscheidung
    final_state = make_final_decision(updated_state, "reject")
    print(f"\nEntscheidung des Fraud Managers: {final_state['fraud_manager_decision'].upper()}")

    return final_state


# Hauptfunktion
def main():
    print("=== FRAUD DETECTION SYSTEM ===")
    print("Verarbeite Transaktion...")

    # Transaktion verarbeiten
    result = process_transaction(example_transaction)

    # Wenn manuelle Überprüfung erforderlich ist
    if result["coordinator_decision"] in ["manual_review", "reject"]:
        final_result = simulate_fraud_manager_interface(result)
    else:
        print(f"\nAutomatische Entscheidung: {result['coordinator_decision'].upper()}")
        final_result = result

    print("\n=== ZUSAMMENFASSUNG ===")
    print(f"Transaktion: {final_result['transaction']['id']}")
    print(f"Ursprüngliche Bewertung: {final_result['coordinator_decision']}")
    if final_result.get('fraud_manager_decision'):
        print(f"Finale Entscheidung: {final_result['fraud_manager_decision']}")
    else:
        print(f"Finale Entscheidung: {final_result['coordinator_decision']}")
    print("Verarbeitung abgeschlossen.")


if __name__ == "__main__":
    main()
