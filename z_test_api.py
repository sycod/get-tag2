import requests


url = "http://get-tag-bc104755b0e2.herokuapp.com/predict"

TITLE_PLACEHOLDER = "example: pandas merge with Python >3.5"
BODY_PLACEHOLDER = """example:
How do I add NaNs for missing rows after a merge?
How do I get rid of NaNs after merging?
I've seen these recurring questions asking about various facets of the pandas merge functionality, the aim here is to collate some of the more important points for posterity."""
user_input = TITLE_PLACEHOLDER + "\n" + BODY_PLACEHOLDER

payload = {"user_input": user_input}
headers = {'Content-Type': 'application/json; charset=utf-8'}
response = requests.post(url, json=payload, headers=headers, timeout=5)

# Vérifier le code de statut de la réponse
if response.status_code == 200:
    # La requête a réussi
    print("Requête POST réussie !")
    print(response.text)  # Afficher le contenu de la réponse
else:
    # La requête a échoué
    print(f"Erreur lors de la requête POST : {response.status_code}")
