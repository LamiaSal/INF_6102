'''
https://www.researchgate.net/publication/325433420_Automatically_Generating_and_Solving_Eternity_II_Style_Puzzles
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.633.8318&rep=rep1&type=pdf
https://www.researchgate.net/publication/267412224_Solving_Eternity-II_puzzles_with_a_tabu_search_algorithm
a = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]]])
python main.py --agent=heuristic --infile=instances/eternity_trivial_B.txt
'''


import numpy as np
import time as t

def solve_heuristic(eternity_puzzle, k):
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :param k: nombre de solutions testées
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    pieces = transform(eternity_puzzle)
    n = eternity_puzzle.board_size
    corners, edges, interiors = split(pieces)
    best_sol = generate_solution(n, corners, edges, interiors)
    best_score = evaluation(best_sol)

    t0 = t.time()
    somme = 0
    for i in range(k-1):
        solution = generate_solution(n, corners, edges, interiors)
        score = evaluation(solution)
        somme += score
        if score < best_score:
            best_sol = solution.copy()
            best_score = score
    solution_final = retransform(best_sol)

    tf =t.time()
    a = open("moyenne_heur_"+eternity_puzzle.instance_file[10:], 'w')
    a.write(" num itération: " + str(k) + "\n")
    a.write(" score total: " + str(somme) + "\n")
    a.write(" score moyen : " + str(somme/k)+ "\n")
    a.write(" temps total: " + str(tf-t0)+ "\n")
    a.write(" temps moyen : " + str((tf-t0)/k))

    a.close()
            
    return (solution_final, best_score)


def generate_solution(n, corners, edges, interiors):
    solution = np.zeros((n,n,4), dtype=np.uint8) #On représente la solution par une matrice 3D
    open_corners = np.full(len(corners), True, dtype=bool)
    open_edges = np.full(len(edges), True, dtype=bool)
    open_interiors = np.full(len(interiors), True, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:  # On fixe le coin en haut à gauche pou éviter les solutions symétriques par rotation
                open_corners[0] = False
                piece = corners[0]
                if piece[0] == 0 and piece[1] == 0:
                    piece = np.roll(piece, -1)
                elif piece[1] == 0 and piece[2] == 0:
                    piece = np.roll(piece, 2)
                elif piece[2] == 0 and piece[3] == 0:
                    piece = np.roll(piece, 1)
                solution[0, 0] = piece
            elif i == 0 and j == n - 1:  # Coin en haut à droite
                left_color = solution[0, n - 2, 1]
                mask1 = (corners[open_corners, 0] == left_color) & (corners[open_corners, 1] == 0)
                mask2 = (corners[open_corners, 1] == left_color) & (corners[open_corners, 2] == 0)
                mask3 = (corners[open_corners, 2] == left_color) & (corners[open_corners, 3] == 0)
                mask4 = (corners[open_corners, 3] == left_color) & (corners[open_corners, 0] == 0)
                mask = mask1 | mask2 | mask3 | mask4
                valid = np.nonzero(mask)[0]
                if len(valid) > 0:
                    piece_i = np.arange(len(corners))[open_corners][np.random.choice(valid)]
                else:
                    piece_i = np.random.choice(np.nonzero(open_corners)[0])
                open_corners[piece_i] = False
                piece = corners[piece_i]
                if piece[1] == 0 and piece[2] == 0:
                    piece = np.roll(piece, -1)
                elif piece[2] == 0 and piece[3] == 0:
                    piece = np.roll(piece, 2)
                elif piece[3] == 0 and piece[0] == 0:
                    piece = np.roll(piece, 1)
                solution[0, n - 1] = piece
            elif i == n - 1 and j == 0:  # Coin en bas à gauche
                up_color = solution[n - 2, 0, 2]
                mask1 = (corners[open_corners, 0] == up_color) & (corners[open_corners, 3] == 0)
                mask2 = (corners[open_corners, 1] == up_color) & (corners[open_corners, 0] == 0)
                mask3 = (corners[open_corners, 2] == up_color) & (corners[open_corners, 1] == 0)
                mask4 = (corners[open_corners, 3] == up_color) & (corners[open_corners, 2] == 0)
                mask = mask1 | mask2 | mask3 | mask4
                valid = np.nonzero(mask)[0]
                if len(valid) > 0:
                    piece_i = np.arange(len(corners))[open_corners][np.random.choice(valid)]
                else:
                    piece_i = np.random.choice(np.nonzero(open_corners)[0])
                open_corners[piece_i] = False
                piece = corners[piece_i]
                if piece[3] == 0 and piece[0] == 0:
                    piece = np.roll(piece, -1)
                elif piece[0] == 0 and piece[1] == 0:
                    piece = np.roll(piece, 2)
                elif piece[1] == 0 and piece[2] == 0:
                    piece = np.roll(piece, 1)
                solution[n - 1, 0] = piece
            elif i == n - 1 and j == n - 1:  # Coin en bas à droite
                # On met simplement le coin restant
                piece_i = np.nonzero(open_corners)[0][0]
                open_corners[piece_i] = False
                piece = corners[piece_i]
                if piece[3] == 0 and piece[2] == 0:
                    piece = np.roll(piece, -1)
                elif piece[0] == 0 and piece[3] == 0:
                    piece = np.roll(piece, 2)
                elif piece[1] == 0 and piece[0] == 0:
                    piece = np.roll(piece, 1)
                solution[n - 1, n - 1] = piece
            elif i == 0:  # Bord haut
                left_color = solution[0, j - 1, 1]
                mask1 = (edges[open_edges, 0] == left_color) & (edges[open_edges, 1] == 0)
                mask2 = (edges[open_edges, 1] == left_color) & (edges[open_edges, 2] == 0)
                mask3 = (edges[open_edges, 2] == left_color) & (edges[open_edges, 3] == 0)
                mask4 = (edges[open_edges, 3] == left_color) & (edges[open_edges, 0] == 0)
                mask = mask1 | mask2 | mask3 | mask4
                valid = np.nonzero(mask)[0]
                if len(valid) > 0:
                    piece_i = np.arange(len(edges))[open_edges][np.random.choice(valid)]
                else:
                    piece_i = np.random.choice(np.nonzero(open_edges)[0])
                open_edges[piece_i] = False
                piece = edges[piece_i]
                if piece[1] == 0:
                    piece = np.roll(piece, -1)
                elif piece[2] == 0:
                    piece = np.roll(piece, 2)
                elif piece[3] == 0:
                    piece = np.roll(piece, 1)
                solution[0, j] = piece
            elif i == n - 1:  # Bord bas
                left_color = solution[n - 1, j - 1, 1]
                up_color = solution[n - 2, j, 2]
                heuristic = np.zeros(sum(open_edges), dtype=np.uint8)
                mask1 = (edges[open_edges, 0] == left_color) & (edges[open_edges, 3] == 0)
                mask2 = (edges[open_edges, 1] == left_color) & (edges[open_edges, 0] == 0)
                mask3 = (edges[open_edges, 2] == left_color) & (edges[open_edges, 1] == 0)
                mask4 = (edges[open_edges, 3] == left_color) & (edges[open_edges, 2] == 0)
                mask = mask1 | mask2 | mask3 | mask4
                heuristic += mask
                mask1 = (edges[open_edges, 3] == 0) & (edges[open_edges, 1] == up_color)
                mask2 = (edges[open_edges, 0] == 0) & (edges[open_edges, 2] == up_color)
                mask3 = (edges[open_edges, 1] == 0) & (edges[open_edges, 3] == up_color)
                mask4 = (edges[open_edges, 2] == 0) & (edges[open_edges, 0] == up_color)
                mask = mask1 | mask2 | mask3 | mask4
                heuristic += mask
                hmax = heuristic.max()
                piece_i = np.arange(len(edges))[open_edges][np.random.choice(np.nonzero(heuristic == hmax)[0])]
                open_edges[piece_i] = False
                piece = edges[piece_i]
                if piece[0] == 0:
                    piece = np.roll(piece, 2)
                elif piece[1] == 0:
                    piece = np.roll(piece, 1)
                elif piece[3] == 0:
                    piece = np.roll(piece, -1)
                solution[n - 1, j] = piece
            elif j == 0:  # Bord gauche
                up_color = solution[i - 1, 0, 2]
                mask1 = (edges[open_edges, 0] == up_color) & (edges[open_edges, 3] == 0)
                mask2 = (edges[open_edges, 1] == up_color) & (edges[open_edges, 0] == 0)
                mask3 = (edges[open_edges, 2] == up_color) & (edges[open_edges, 1] == 0)
                mask4 = (edges[open_edges, 3] == up_color) & (edges[open_edges, 2] == 0)
                mask = mask1 | mask2 | mask3 | mask4
                valid = np.nonzero(mask)[0]
                if len(valid) > 0:
                    piece_i = np.arange(len(edges))[open_edges][np.random.choice(valid)]
                else:
                    piece_i = np.random.choice(np.nonzero(open_edges)[0])
                open_edges[piece_i] = False
                piece = edges[piece_i]
                if piece[0] == 0:
                    piece = np.roll(piece, -1)
                elif piece[1] == 0:
                    piece = np.roll(piece, 2)
                elif piece[2] == 0:
                    piece = np.roll(piece, 1)
                solution[i, 0] = piece
            elif j == n - 1:  # Bord droit
                left_color = solution[i, n - 2, 1]
                up_color = solution[i - 1, n - 1, 2]
                heuristic = np.zeros(sum(open_edges), dtype=np.uint8)
                mask1 = (edges[open_edges, 0] == left_color) & (edges[open_edges, 2] == 0)
                mask2 = (edges[open_edges, 1] == left_color) & (edges[open_edges, 3] == 0)
                mask3 = (edges[open_edges, 2] == left_color) & (edges[open_edges, 0] == 0)
                mask4 = (edges[open_edges, 3] == left_color) & (edges[open_edges, 1] == 0)
                mask = mask1 | mask2 | mask3 | mask4
                heuristic += mask
                mask1 = (edges[open_edges, 3] == 0) & (edges[open_edges, 2] == up_color)
                mask2 = (edges[open_edges, 0] == 0) & (edges[open_edges, 3] == up_color)
                mask3 = (edges[open_edges, 1] == 0) & (edges[open_edges, 0] == up_color)
                mask4 = (edges[open_edges, 2] == 0) & (edges[open_edges, 1] == up_color)
                mask = mask1 | mask2 | mask3 | mask4
                heuristic += mask
                hmax = heuristic.max()
                piece_i = np.arange(len(edges))[open_edges][np.random.choice(np.nonzero(heuristic == hmax)[0])]
                open_edges[piece_i] = False
                piece = edges[piece_i]
                if piece[2] == 0:
                    piece = np.roll(piece, -1)
                elif piece[3] == 0:
                    piece = np.roll(piece, 2)
                elif piece[0] == 0:
                    piece = np.roll(piece, 1)
                solution[i, n - 1] = piece
            else:  # Intérieur
                left_color = solution[i, j - 1, 1]
                up_color = solution[i - 1, j, 2]
                heuristic = np.zeros(sum(open_interiors), dtype=np.uint8)
                mask1 = (interiors[open_interiors, 0] == left_color) | (interiors[open_interiors, 0] == up_color)
                mask2 = (interiors[open_interiors, 1] == left_color) | (interiors[open_interiors, 1] == up_color)
                mask3 = (interiors[open_interiors, 2] == left_color) | (interiors[open_interiors, 2] == up_color)
                mask4 = (interiors[open_interiors, 3] == left_color) | (interiors[open_interiors, 3] == up_color)
                mask = mask1 | mask2 | mask3 | mask4
                heuristic += mask
                mask1 = (interiors[open_interiors, 3] == left_color) & (interiors[open_interiors, 0] == up_color)
                mask2 = (interiors[open_interiors, 0] == left_color) & (interiors[open_interiors, 1] == up_color)
                mask3 = (interiors[open_interiors, 1] == left_color) & (interiors[open_interiors, 2] == up_color)
                mask4 = (interiors[open_interiors, 2] == left_color) & (interiors[open_interiors, 3] == up_color)
                mask = mask1 | mask2 | mask3 | mask4
                heuristic += mask
                hmax = heuristic.max()
                piece_i = np.arange(len(interiors))[open_interiors][np.random.choice(np.nonzero(heuristic == hmax)[0])]
                open_interiors[piece_i] = False
                piece = interiors[piece_i]
                if hmax == 2:
                    if piece[1] == up_color:
                        piece = np.roll(piece, -1)
                    elif piece[2] == up_color:
                        piece = np.roll(piece, 2)
                    elif piece[3] == up_color:
                        piece = np.roll(piece, 1)
                elif hmax == 1:
                    if piece[1] == up_color:
                        piece = np.roll(piece, -1)
                    elif piece[2] == up_color:
                        piece = np.roll(piece, 2)
                    elif piece[3] == up_color:
                        piece = np.roll(piece, 1)
                    elif piece[0] == left_color:
                        piece = np.roll(piece, -1)
                    elif piece[1] == left_color:
                        piece = np.roll(piece, 2)
                    elif piece[2] == left_color:
                        piece = np.roll(piece, 1)
                solution[i, j] = piece
    return solution


def transform(eternity_puzzle):
    """
    Par défaut les pièces sont des lignes dans l'ordre N>S>W>E.
    Réorgansiation des pièces dans un array numpy avec un sens horaire des couleurs N>E>S>W pour roll plus rapidement les tableaux.
    :param eternity_puzzle:
    :return:
    """
    pieces = np.array(eternity_puzzle.piece_list, dtype=np.uint8)
    pieces = pieces[:,[0,3,1,2]]
    return pieces


def retransform(solution):
    """
    retransforme une solution dans notre format 3D numpy en une liste de pièces d'en bas à gauche vers en haut à doite ligne par ligne avec l'odre d'origine N>S>W>E.
    :param solution:
    :return:
    """
    n = len(solution)
    list_sol = solution[::-1].reshape((n*n,4))[:,[0,2,3,1]].tolist()
    return [tuple(x) for x in list_sol]


def split(pieces):
    n = int(np.sqrt(len(pieces)))

    counts = np.sum(pieces == 0,axis=1)

    corners = pieces[np.nonzero(counts == 2)]
    edges = pieces[np.nonzero(counts == 1)] #FIXME : Ne marche pas avec l'instance trivial A a priori
    interiors = pieces[np.nonzero(counts == 0)]

    assert len(corners) == 4, "Il n'y a pas 4 coins"
    assert len(edges) == (n-2)*4, "Il n'y a pas (n-2)*4 arrêtes"
    assert len(interiors) == (n-2)**2, "Il n'y a pas (n-2)**2 pièces intérieures"

    return corners, edges, interiors


def evaluation(solution):
    n = len(solution)
    vertical = solution[:n-1,:,2] != solution[1:,:,0]
    horizontal = solution[:,:n-1,1] != solution[:,1:,3]
    top = solution[0, :, 0] != 0
    bot = solution[n-1, :, 2] != 0
    right = solution[:, n-1, 1] != 0
    left = solution[:, 0, 3] != 0
    score = vertical.sum() + horizontal.sum() + top.sum() + bot.sum() + right.sum() + left.sum()
    return score
