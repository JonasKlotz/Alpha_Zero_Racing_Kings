# Implementierung von Alpha Zero für das AOT KI-Projekt SoSe20

## Zielstellung

Wir implementieren den Alpha-Zero-Algorithmus als Künstliche Intelligenz für das Spiel _Racing Kings_.
* Die KI lernt anhand der Spielregeln, die in Form eines Zuggenerators zur Verfügung gestellt wird
* Ein Übersetzungsmodul übersetzt Züge in "menschlicher" Notation (FEN) in [Tensor-Notation](glossary#tensor-notation) (ähnlich wie im [deepmind paper](https://arxiv.org/pdf/1712.01815.pdf))
* Mit einem spielspezifischem Zuggenerator und Übersetzer kann die KI _prinzipiell_ alle Spiele erlernen
* Dieses Projekt konzentriert sich jedoch auf das Erlernen von _Racing Kings_.
* Die KI kommuniziert Züge im Spielbetrieb über einen Port in FEN-Notation
* Die Port-Schnittstelle ist so gestaltet, dass die KI über den Game-Server am Turnier teilnehmen kann
* Das zugrundeliegende Neuronale Netzwerk ist klein genug dimensioniert, dass effektives Training und Testen mit den gegebenen Ressourcen möglich ist

## Zusammenfassung des Algorithmus

* Monte Carlo Tree Search (_MCTS_) probiert bei der Berechnung eines Zugs $`n`$ zufällige Spielverläufe bis zum Spielende und erstellt daraus eine Verteilung $`\pi`$, welche Spielzüge am ehesten zum Sieg führen
* Alpha Zero Tree Search wählt im Gegensatz zu _MCTS_ Züge, die aufgrund der Spielstellung $`s`$ von einer Bewertungsfunktion $`f_\theta`$ als vielversprechend gewertet werden
* Um die Bewertungen von $`f`$ zu verbessern, müssen die Parameter $`\theta`$ an einem Datensatz mit realistischen Spieldaten trainiert werden
* Insbesondere ist für jeden Spielzug der tatsächliche Ausgang der entsprechenden Partie $`z \in \{-1, 0, 1\}`$ relevant
* Ein Datenpunkt im Datensatz besteht aus $`(s, \pi, z)`$: Spielstellung, Zugverteilung nach Alpha Zero Tree Search, Ausgang der Partie
* Im Training wird ein Datensatz durch Spiele gegen sich selbst erstellt und im Anschluss das Modell $`f_\theta`$ durch Gradientenabstieg auf dem Datensatz trainiert; danach wird mit dem neuen, besseren Modell ein Datensatz durch Spiele gegen sich selbst erstellt und das Modell auf dem neuen Datensatz trainiert und so weiter
* Ziel des Trainings ist, $`z`$ und $`\pi`$ aus $`s`$ vorherzusagen und somit aus der zufälligen _MCTS_ eine bewertungsgeleitete Alpha Zero Tree Search zu machen
* Die Originalimplementierung nutzt hierfür _ein_ Modell mit zwei _heads_, also $`p, v = f_\theta (s)`$, wobei im Training der Fehler sowohl zwischen $`p`$ und $`\pi`$ (_policy_, Zugempfehlungen) minimiert wird als auch zwischen $`v`$ und $`z`$ (_value_, Stellungsbewertung)
* Falls das Modell nicht hinreichend trainiert ist, können im Spielbetrieb illegale Züge vorgeschlagen werden!



## Zeitplan
In den Arbeitsbereichen soll möglichst parallel gearbeitet werden. Dazu stellen die einzelnen Gruppen den anderen möglichst früh _mock up_-Objekte zur Verfügung, die bereits die Schnittstelle implementieren, wenn auch mit eingeschränkter oder suboptimaler Funktionalität. [Zeitplan als PDF](report/projektplan.pdf).





KW | SW | Projekt | Organisation | MCTS | Modell | Interface | Testen
--- | --- | --- | --- | --- | --- | --- | --- 
18 |  2 | Recherche | Aufteilung der Arbeitspakete | | [Recherche](modell#material) | | 
19 |  3 | Abschluss Planungsphase | | [Detaillierter Entwurfsplan](mcts#detaillierter-entwurfsplan) | [Architekturentwurf](modell#architekturentwurf), [Zugrepräsentation (Tensor-Notation)](modell#zugrepräsentation) und [Entwicklungsstrategie (wie testen?)](modell#entwicklungsstrategie) | [Zuggenerator implementieren](interface#zuggenerator) | Tests für Zuggenerator anhand Sequenzdiagramme entwerfen
20 |  4 | |  | Implementierung und [ersten Datensatz erstellen](mcts#ersten-datensatz-erstellen) | [Infrastruktur (load, configure) implementieren](modell#infrastruktur) und [auf erstem Datensatz trainieren](modell#erstes-training) | [Übersetzer FEN-Tensor-Notation implementieren](interface#Übersetzer), [Funktion für Spielbetrieb implementieren](interface#spielbetrieb) | MCTS-Tests anhand Entwurf schreiben, Übersetzer-Tests anhand Sequenzdiagramme schreiben
21 |  5 | | | Optimierung und Testen | Modell-Training im Lernbetrieb | [Funktion für Lernbetrieb implementieren](interface#lernbetrieb) | Modultests
22 |  6 | | Evaluation und weitere Planung des Modelltrainings | Zusammenführung mit Modell, Evaluation | Experimentelles Training: Hyperparameter variieren | ELO-Funktion implementieren | Integrationstests
23 |  7 | |  | | | [Port-Interface passend zum Game-Server implementieren](interface#port-interface) | 
24 |  8 | funktionsfähiger Prototyp | | | | | 
25 |  9 | | | | | | 
26 | 10 | | | | | | 
27 | 11 | | | | | | 
28 | 12 | | | | | | 
29 | 13 | Teilnahme am KI-Turnier, Abschlusspräsentation | | | | | 
20 | 14 | | | | | | 
21 | 15 | | | | | | 



