'''
https://www.researchgate.net/publication/325433420_Automatically_Generating_and_Solving_Eternity_II_Style_Puzzles
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.633.8318&rep=rep1&type=pdf
https://www.researchgate.net/publication/267412224_Solving_Eternity-II_puzzles_with_a_tabu_search_algorithm
'''


import numpy as np

def solve_heuristic(eternity_puzzle):
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    pieces = transform(eternity_puzzle)
    n = len(pieces)
    solution = np.zeroes((n,n,4), dtype=np.uint8) #On représente la solution par une matrice 3D
    corners, edges, interiors = split(pieces)
    open_corners = np.full(len(corners),True)
    open_edges = np.full(len(edges),True)
    open_interiors = np.full(len(interiors),True)
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0: # On fixe le coin en haut à gauche pou éviter les solutions symétriques par rotation
                open_corners[0] = False
                piece = corners[0]
                if piece[0] == 0 and piece[1] == 0:
                    piece = np.roll(piece, -1)
                elif piece[1] == 0 and piece[2] == 0:
                    piece = np.roll(piece, 2)
                elif piece[2] == 0 and piece[3] == 0:
                    piece = np.roll(piece, 1)
                solution[0,0] == piece
            elif i == 0 and j == n-1: # Coin en haut à droite
                left_color = solution[0,n-2,1]
                mask1 = (corners[open_corners, 0] == left_color) & (corners[open_corners, 1] == 0)
                mask2 = (corners[open_corners, 1] == left_color) & (corners[open_corners, 2] == 0)
                mask3 = (corners[open_corners, 2] == left_color) & (corners[open_corners, 3] == 0)
                mask4 = (corners[open_corners, 3] == left_color) & (corners[open_corners, 0] == 0)
                mask = mask1 | mask2 | mask3 | mask4
                piece_i = np.arange(len(corners))[open_corners][np.random.choice(np.nonzero(mask)[0])]
                open_corners[piece_i] = False
                piece = corners[piece_i]
                if piece[1] == 0 and piece[2] == 0:
                    piece = np.roll(piece, -1)
                elif piece[2] == 0 and piece[3] == 0:
                    piece = np.roll(piece, 2)
                elif piece[3] == 0 and piece[0] == 0:
                    piece = np.roll(piece, 1)
                solution[0, n-1] == piece
            elif i == n-1  and j == 0: # Coin en bas à gauche
                up_color = solution[n - 2, 0, 2]
                mask1 = (corners[open_corners, 0] == up_color) & (corners[open_corners, 3] == 0)
                mask2 = (corners[open_corners, 1] == up_color) & (corners[open_corners, 0] == 0)
                mask3 = (corners[open_corners, 2] == up_color) & (corners[open_corners, 1] == 0)
                mask4 = (corners[open_corners, 3] == up_color) & (corners[open_corners, 2] == 0)
                mask = mask1 | mask2 | mask3 | mask4
                piece_i = np.arange(len(corners))[open_corners][np.random.choice(np.nonzero(mask)[0])]
                open_corners[piece_i] = False
                piece = corners[piece_i]
                if piece[3] == 0 and piece[0] == 0:
                    piece = np.roll(piece, -1)
                elif piece[0] == 0 and piece[1] == 0:
                    piece = np.roll(piece, 2)
                elif piece[1] == 0 and piece[2] == 0:
                    piece = np.roll(piece, 1)
                solution[0, n - 1] == piece
            elif i == n-1 and j == n-1: # Coin en bas à droite
                # On met simplement le coin restant
                piece_i = np.nonzero(open_corners)[0][0]
                open_corners[piece_i] = False
                solution[n-1, n-1] == corners[piece_i]
            elif i == 0: # Bord haut
                pass
            elif i == n-1: # Bord bas
                pass
            elif j == 0: # Bord gauche
                pass
            elif j == n-1: # Bord droit
                pass
            else: # Intérieur
                pass

    return


def transform(eternity_puzzle):
    """
    Par défaut les pièces sont des lignes dans l'ordre N>S>W>E.
    Réorgansiation des pièces dans un array numpy avec un sens horaire des couleurs N>E>S>W pour roll plus rapidement les tableaux.
    :param eternity_puzzle:
    :return:
    """
    n = eternity_puzzle.board_size
    pieces = np.zeroes((n**2,4), dtype=np.uint8)
    pieces_list = eternity_puzzle.piece_list
    for i,u in enumerate(pieces_list):
        for j,v in enumerate(u):
            if j==0:
                pieces[i,0] = v
            elif j==1:
                pieces[i, 2] = v
            elif j==2:
                pieces[i, 3] = v
            else:
                pieces[i, 1] = v
    return pieces


def retransform(solution):
    """
    retransforme une solution dans notre format 3D numpy en une liste de pièces d'en bas à gauche vers en haut à doite ligne par ligne avec l'odre d'origine N>S>W>E.
    :param solution:
    :return:
    """
    return


def split(pieces):
    n = len(pieces)

    counts = np.sum(pieces == 0,axis=1)

    corners = pieces[np.nonzero(counts == 2)]
    edges = pieces[np.nonzero(counts == 1)] #FIXME : Ne marche pas avec l'instance trivial A a priori
    interiors = pieces[np.nonzero(counts == 0)]

    assert len(corners) == 4, "Il n'y a pas 4 coins"
    assert len(edges) == (n-2)*4, "Il n'y a pas (n-2)*4 arrêtes"
    assert len(interiors) == (n-2)**2, "Il n'y a pas (n-2)**2 pièces intérieures"

    return corners, edges, interiors


def evaluation(solution):
    return
