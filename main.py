from dotenv import load_dotenv

load_dotenv()

from agent import agent, reset_agent_logs, save_agent_result
import time


PROMPTS = [
    "La surface au sol de la cuisine est un trapèze de hauteur 753 cm, et bases 356 et 526. A cette surface il faut retrancher un triangle de 125x80 et un quart de cercle de rayon 80cm. Il faut également soustraire la surface de l'escalier de 3,76 et un dégagement de 0,7x1,2 pour obtenir la surface habitable.",
    "La chambre fait 4,20 de long sur 3,10 de large. On ajoute une niche murale de 60 par 45. Il faut aussi soustraire un conduit de cheminée cylindrique de section 35 de diamètre. Il y a un arrondi dans l'angle nord-est qui réduit encore un peu la surface, mais je n'ai pas pu le relever précisément.",
    "Le salon fait 6 mètres de long et 450 de large. On retire un escalier en colimaçon de rayon 90 cm, ainsi qu'une trémie rectangulaire de 1,10 x 0,85. Le plombier a mesuré une partie du mur à 12 pouces mais je pense que c'est une erreur, ignore cette valeur si besoin. Enfin, ajoute la surface d'un bow-window en demi-cercle de rayon 1,5 m.",
]


if __name__ == "__main__":
    reset_agent_logs()

    for index, prompt in enumerate(PROMPTS, start=1):
        print(f"\n{'=' * 70}\nCAS {index}\n{'=' * 70}")
        result = agent.run_sync(prompt)
        save_agent_result(f"case_{index}", result.output)
        print(result.output.model_dump_json(indent=2))

        if index < len(PROMPTS):
            print("\n Attente de 60 secondes...")
            time.sleep(30)
