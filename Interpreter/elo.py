# Je genauer wir die Alpha Elo schätzen desto schneller Konvergiert Die alpha Elo
# Meine Idee wir setzen SF Elo fest und lassen alpha einmal konvergieren gegen aufsteigende fische


def calculate_elo(R_a, R_b, S_a, S_b, k=20):
    """

    :param k: Faktor für Eloanstieg, k ist üblicherweise 20, bei Top-Spielern (Elo > 2400) 10, bei weniger als 30 gewerteten Partien 40,
    :param R_a: Elo von A
    :param R_b: Elo von B
    :param S_a: Tatsächlicher Sieg für a ,b. Remi = 0.5
    :param S_b: Tatsächlicher Sieg für a ,b. Remi = 0.5
    :return: neue Elos für A,B
    """

    k = 20
    R_bb = R_a - R_b
    R_aa = R_b - R_a
    # Normalerweise wird ab 400 Punkte differenz eine Grenze gezogen, um nicht zu schnell elo zu gewinnen/verlieren
    # Das könnten wir weglassen um die Konversion zu beschleunigen
    if abs(R_a) > 400:
        if R_a > R_b:
            R_aa = -400  # spieler mit weniger negativ
            R_bb = 400  # spieler mit mehr positiv

        else:
            R_aa = 400  # spieler mit mehr positiv
            R_bb = -400  # spieler mit weniger negativ

    # Erwartete Punktzahl, P(Win_a) + P(Rem_a)
    E_a = 1 / (1 + pow(10, (R_aa) / 400))
    E_b = 1 / (1 + pow(10, (R_bb) / 400))

    R_a_new = round(R_a + k * (S_a - E_a))
    R_b_new = round(R_b + k * (S_b - E_b))

    return R_a_new, R_b_new
