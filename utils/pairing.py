"""Quali coppie di immagini vale la pena matchare.

Una sola regola sostituisce le tre euristiche separate usate in passato -- coppie
consecutive, coppie a salto uno, e ricerca a raggio fisso fra passate adiacenti: due
scatti si accoppiano se le loro impronte a terra si sovrappongono abbastanza. La stessa
soglia governa il legame lungo la passata e quello fra passate, perche' e' la stessa
domanda posta due volte.

Il tetto sui vicini serve a non sprecare: con overlap frontale all'80% ogni scatto ha una
dozzina di partner plausibili, e i piu' deboli non aggiungono informazione che i piu'
forti non diano gia'. Si tengono i migliori per sovrapposizione.
"""
import numpy as np

from utils.footprint import centers_and_radii, overlap_fraction


def candidate_pairs(
    quads: np.ndarray,
    min_overlap: float = 0.25,
    max_neighbors: int = 6,
) -> list[tuple[int, int, float]]:
    """Coppie (i, j, sovrapposizione) con i < j, ordinate per indice.

    `quads` e' l'array (N, 4, 2) delle impronte, nello stesso ordine delle pose. Una coppia
    sopravvive se supera `min_overlap` ED e' fra i `max_neighbors` partner migliori di
    almeno uno dei due scatti: basta che uno dei due la consideri utile perche' il legame
    valga la pena.
    """
    n = len(quads)
    if n < 2:
        return []

    centers, radii = centers_and_radii(quads)
    sovrapposizioni: dict[tuple[int, int], float] = {}

    for i in range(n - 1):
        # Prefiltro: oltre la somma dei raggi le impronte non possono toccarsi.
        distanze = np.linalg.norm(centers[i + 1 :] - centers[i], axis=1)
        vicini = np.nonzero(distanze < radii[i] + radii[i + 1 :])[0] + i + 1
        for j in vicini:
            frazione = overlap_fraction(quads[i], quads[int(j)])
            if frazione >= min_overlap:
                sovrapposizioni[(i, int(j))] = frazione

    if not sovrapposizioni:
        return []

    # Tetto sui vicini, applicato dal punto di vista di ciascuno dei due scatti.
    per_frame: dict[int, list[tuple[float, tuple[int, int]]]] = {}
    for coppia, frazione in sovrapposizioni.items():
        for estremo in coppia:
            per_frame.setdefault(estremo, []).append((frazione, coppia))

    tenute: set[tuple[int, int]] = set()
    for candidate in per_frame.values():
        candidate.sort(key=lambda x: -x[0])
        for _, coppia in candidate[:max_neighbors]:
            tenute.add(coppia)

    return sorted((i, j, sovrapposizioni[(i, j)]) for i, j in tenute)


def connected_components(n: int, pairs) -> list[list[int]]:
    """Componenti connesse del grafo delle coppie, dalla piu' grande alla piu' piccola.

    Senza GPS il grafo e' l'unica cosa che tiene insieme le passate: se si spezza in piu'
    componenti, i pezzi non hanno alcun legame reciproco e le loro posizioni relative
    restano indeterminate. Va controllato PRIMA di spendere il matching e l'ottimizzazione,
    perche' e' un guasto che nessun peso puo' compensare.
    """
    padre = list(range(n))

    def radice(x: int) -> int:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for coppia in pairs:
        a, b = coppia[0], coppia[1]
        ra, rb = radice(a), radice(b)
        if ra != rb:
            padre[ra] = rb

    gruppi: dict[int, list[int]] = {}
    for i in range(n):
        gruppi.setdefault(radice(i), []).append(i)
    return sorted(gruppi.values(), key=len, reverse=True)


def summarize(n: int, pairs) -> dict:
    """Statistiche del grafo, da stampare prima di lanciare il matching."""
    componenti = connected_components(n, pairs)
    gradi = np.zeros(n, dtype=int)
    for coppia in pairs:
        gradi[coppia[0]] += 1
        gradi[coppia[1]] += 1
    return {
        "pairs": len(pairs),
        "components": len(componenti),
        "largest_component": len(componenti[0]) if componenti else 0,
        "isolated": int((gradi == 0).sum()),
        "degree_min": int(gradi.min()) if n else 0,
        "degree_median": float(np.median(gradi)) if n else 0.0,
        "degree_max": int(gradi.max()) if n else 0,
    }
