import numpy as np
import time
import solver_heuristic_greedy as shg
#content = np.loadtxt("sample.txt")

def solve_advanced(eternity_puzzle, tag, duration):
    #TODO : Ajouter listing pour tracé de courbe
    #TODO : Ajouter mémoire des restarts pour la moyenne globale
    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    start_time = time.time()

    n = eternity_puzzle.board_size

    max_duration = duration  # Temps alloué
    max_iter_border = int(200 * 16/n)
    patience_for_SA = int(100 * 16/n)
    patience_for_ILS = int(250 * 16/n)
    t0 = 100
    cooling = 0.99
    perturbation_ratio = 0.5

    pieces = transform(eternity_puzzle)
    corners, edges, interiors = split(pieces)
    tabu_length = int(1.5 * n * n /4)

    solution, score = init_greedy(1000, n, corners, edges, interiors)
    print("Init finie, score de départ : ", score)
    best_sol = solution.copy()
    best_score = score
    temp_best_score = score
    border_iter = 0
    inside_iter_tabu = 0
    inside_iter_SA = 0
    ILS_iter = 0
    best_score_mem = [best_score]
    best_score_timestamp = [0]
    best_score_at_restart = [best_score]
    while time.time() < max_duration + start_time and best_score > 0:
        ILS_iter += 1
        solution, nb_iter = solve_border(solution, max_iter_border, max_duration, start_time, tabu_length)
        border_iter += nb_iter
        score = evaluation(solution)
        if score < temp_best_score:
            temp_best_score = score

        if score < best_score:
            best_score_mem.append(score)
            best_score_timestamp.append(round((time.time() - start_time),2))
            best_sol = solution.copy()
            best_score = score

        if time.time() > max_duration + start_time or best_score == 0:
            break

        not_stagnating = True
        while not_stagnating:
            solution, score, nb_iter = solve_inside(solution, tabu_length, temp_best_score, patience_for_SA, max_duration, start_time)
            inside_iter_tabu += nb_iter
            if score < temp_best_score:
                temp_best_score = score

            if score < best_score:
                best_score_mem.append(score)
                best_score_timestamp.append(round((time.time() - start_time), 2))
                best_sol = solution.copy()
                best_score = score

            if time.time() > max_duration + start_time or best_score == 0:
                print("Nb of border iter : ", border_iter)
                print("Nb of tabu iter : ", inside_iter_tabu)
                print("Nb of SA iter : ", inside_iter_SA)
                print("Nb of ILS iter : ", ILS_iter)
                solution_final = retransform(best_sol)
                best_score_mem.append(best_score)
                best_score_timestamp.append(round((time.time() - start_time), 2))
                best_score_at_restart.append(temp_best_score)
                np.savetxt("scores_at_restart"+str(tag)+".txt", np.array(best_score_at_restart), delimiter=", ")
                np.savetxt("best_score_mem"+str(tag)+".txt", np.array(best_score_mem), delimiter=", ")
                np.savetxt("best_score_time"+str(tag)+".txt", np.array(best_score_timestamp), delimiter=", ")
                return solution_final, best_score

            solution, score, improvement, nb_iter = simulated_annealing(solution, t0, cooling, patience_for_ILS, max_duration, start_time)
            inside_iter_SA += nb_iter
            if improvement:
                if score < temp_best_score:
                    temp_best_score = score

                if score < best_score:
                    best_score_mem.append(score)
                    best_score_timestamp.append(round((time.time() - start_time), 2))
                    best_sol = solution.copy()
                    best_score = score
            else:
                not_stagnating = False

            if time.time() > max_duration + start_time or best_score == 0:
                print("Nb of border iter : ", border_iter)
                print("Nb of tabu iter : ", inside_iter_tabu)
                print("Nb of SA iter : ", inside_iter_SA)
                print("Nb of ILS iter : ", ILS_iter)
                solution_final = retransform(best_sol)
                best_score_mem.append(best_score)
                best_score_timestamp.append(round((time.time() - start_time), 2))
                best_score_at_restart.append(temp_best_score)
                np.savetxt("scores_at_restart"+str(tag)+".txt", np.array(best_score_at_restart), delimiter=", ")
                np.savetxt("best_score_mem"+str(tag)+".txt", np.array(best_score_mem), delimiter=", ")
                np.savetxt("best_score_time"+str(tag)+".txt", np.array(best_score_timestamp), delimiter=", ")
                return solution_final, best_score

        solution = perturbation(solution, perturbation_ratio)
        best_score_at_restart.append(temp_best_score)
        temp_best_score = 1000

    print("Nb of border iter : ", border_iter)
    print("Nb of tabu iter : ", inside_iter_tabu)
    print("Nb of SA iter : ", inside_iter_SA)
    print("Nb of ILS iter : ", ILS_iter)
    solution_final = retransform(best_sol)
    best_score_mem.append(best_score)
    best_score_timestamp.append(round((time.time() - start_time), 2))
    best_score_at_restart.append(temp_best_score)
    np.savetxt("scores_at_restart"+str(tag)+".txt", np.array(best_score_at_restart), delimiter=", ")
    np.savetxt("best_score_mem"+str(tag)+".txt", np.array(best_score_mem), delimiter=", ")
    np.savetxt("best_score_time"+str(tag)+".txt", np.array(best_score_timestamp), delimiter=", ")
    return solution_final, best_score


########################################################################################################################
########################################################################################################################
###########                                    Utilitaires généraux                                         ############
########################################################################################################################
########################################################################################################################


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

    counts = np.sum(pieces == 0, axis=1)

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


########################################################################################################################
########################################################################################################################
###########                                       Initialisation                                            ############
########################################################################################################################
########################################################################################################################
def init_semi_random(k, n, corners, edges, interiors):
    best_sol = None
    best_score = 1000
    for i in range(k - 1):
        solution = np.zeros((n, n, 4), dtype=np.uint8)  # On représente la solution par une matrice 3D

        # Placement des sommets
        # On fixe le coin en haut à gauche pou éviter les solutions symétriques par rotation
        piece = corners[0]
        if piece[0] == 0 and piece[1] == 0:
            piece = np.roll(piece, -1)
        elif piece[1] == 0 and piece[2] == 0:
            piece = np.roll(piece, 2)
        elif piece[2] == 0 and piece[3] == 0:
            piece = np.roll(piece, 1)
        solution[0, 0] = piece

        index = np.random.choice(3, size=3, replace=False) + 1
        # Coin en haut à droite
        piece_i = index[0]
        piece = corners[piece_i]
        if piece[1] == 0 and piece[2] == 0:
            piece = np.roll(piece, -1)
        elif piece[2] == 0 and piece[3] == 0:
            piece = np.roll(piece, 2)
        elif piece[3] == 0 and piece[0] == 0:
            piece = np.roll(piece, 1)
        solution[0, n - 1] = piece

        # Coin en bas à gauche
        piece_i = index[1]
        piece = corners[piece_i]
        if piece[3] == 0 and piece[0] == 0:
            piece = np.roll(piece, -1)
        elif piece[0] == 0 and piece[1] == 0:
            piece = np.roll(piece, 2)
        elif piece[1] == 0 and piece[2] == 0:
            piece = np.roll(piece, 1)
        solution[n - 1, 0] = piece

        # Coin en bas à droite
        piece_i = index[2]
        piece = corners[piece_i]
        if piece[3] == 0 and piece[2] == 0:
            piece = np.roll(piece, -1)
        elif piece[0] == 0 and piece[3] == 0:
            piece = np.roll(piece, 2)
        elif piece[1] == 0 and piece[0] == 0:
            piece = np.roll(piece, 1)
        solution[n - 1, n - 1] = piece

        # Placement des bords
        index = np.random.choice(len(edges), size=len(edges), replace=False)
        ind = 0
        # Bord haut
        for i in range(1,n-1):
            piece_i = index[ind]
            piece = edges[piece_i]
            if piece[1] == 0:
                piece = np.roll(piece, -1)
            elif piece[2] == 0:
                piece = np.roll(piece, 2)
            elif piece[3] == 0:
                piece = np.roll(piece, 1)
            solution[0, i] = piece
            ind += 1

        # Bord bas
        for i in range(1,n-1):
            piece_i = index[ind]
            piece = edges[piece_i]
            if piece[0] == 0:
                piece = np.roll(piece, 2)
            elif piece[1] == 0:
                piece = np.roll(piece, 1)
            elif piece[3] == 0:
                piece = np.roll(piece, -1)
            solution[n - 1, i] = piece
            ind += 1

        # Bord gauche
        for i in range(1,n-1):
            piece_i = index[ind]
            piece = edges[piece_i]
            if piece[0] == 0:
                piece = np.roll(piece, -1)
            elif piece[1] == 0:
                piece = np.roll(piece, 2)
            elif piece[2] == 0:
                piece = np.roll(piece, 1)
            solution[i, 0] = piece
            ind += 1

        # Bord droit
        for i in range(1,n-1):
            piece_i = index[ind]
            piece = edges[piece_i]
            if piece[2] == 0:
                piece = np.roll(piece, -1)
            elif piece[3] == 0:
                piece = np.roll(piece, 2)
            elif piece[0] == 0:
                piece = np.roll(piece, 1)
            solution[i, n - 1] = piece
            ind += 1

        # Placement des pièces intérieures
        index = np.random.choice(len(interiors), size=len(interiors), replace=False)
        ind = 0
        for i in range(1,n-1):
            for j in range(1,n-1):
                piece_i = index[ind]
                piece = interiors[piece_i]
                roll = np.random.randint(0, 4)
                piece = np.roll(piece, roll)
                solution[i, j] = piece
                ind += 1

        score = evaluation(solution)
        if score < best_score:
            best_sol = solution.copy()
            best_score = score

    return best_sol, best_score


def init_greedy(k, n, corners, edges, interiors):
    best_sol = shg.generate_solution(n, corners, edges, interiors)
    best_score = evaluation(best_sol)
    for i in range(k - 1):
        solution = shg.generate_solution(n, corners, edges, interiors)
        score = evaluation(solution)
        if score < best_score:
            best_sol = solution.copy()
            best_score = score
    return best_sol, best_score

########################################################################################################################
########################################################################################################################
###########                                     Traitement du bord                                          ############
########################################################################################################################
########################################################################################################################


def solve_border(solution, max_iter, max_duration, start_time, tabu_length):
    n = len(solution)

    score = eval_border(solution)
    best_sol = solution.copy()
    best_score = score
    nb_iter = 0

    tabu_dict = dict()

    delta_matrix = np.zeros(((n-1)*4,(n-1)*4), dtype=np.int8)
    for p1 in range((n-1)*4):
        for p2 in range(p1):
            if p1 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3] or p2 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3]:
                if not (p1 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3] and p2 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3]):
                    continue
                else:
                    delta_matrix[p1, p2] = eval_delta_corners(solution, p1, p2)
            else:
                delta_matrix[p1,p2] = eval_delta_edges(solution, p1, p2)

    while nb_iter < max_iter and time.time() < max_duration + start_time and best_score > 0:
        nb_iter += 1

        best_moves = []
        best_cost = 10
        for p1 in range((n - 1) * 4):
            for p2 in range(p1):
                if p1 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3] or p2 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3]:
                    if not (p1 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3] and p2 in [0, (n - 1), (n - 1) * 2,(n - 1) * 3]):
                        continue
                temp_cost = delta_matrix[p1, p2]
                if score + temp_cost < best_score:
                    if temp_cost < best_cost:
                        best_cost = temp_cost
                        best_moves = [(p1, p2)]
                    elif temp_cost == best_cost:
                        best_moves.append((p1, p2))
                elif (p1,p2) not in tabu_dict or tabu_dict[(p1,p2)] + tabu_length < nb_iter:
                    if temp_cost < best_cost:
                        best_cost = temp_cost
                        best_moves = [(p1, p2)]
                    elif temp_cost == best_cost:
                        best_moves.append((p1, p2))

        p1, p2 = best_moves[np.random.randint(len(best_moves))]
        tabu_dict[(p1, p2)] = nb_iter

        if p1 < n - 1:
            side1 = 0
            p1_i = 0
            p1_j = p1
        elif p1 < (n - 1) * 2:
            side1 = 1
            p1_i = p1 % (n - 1)
            p1_j = n - 1
        elif p1 < (n - 1) * 3:
            side1 = 2
            p1_i = n - 1
            p1_j = (n-1) - (p1 % (n - 1))
        else:
            side1 = 3
            p1_i = (n-1) - (p1 % (n - 1))
            p1_j = 0
        if p2 < n - 1:
            side2 = 0
            p2_i = 0
            p2_j = p2
        elif p2 < (n - 1) * 2:
            side2 = 1
            p2_i = p2 % (n - 1)
            p2_j = n - 1
        elif p2 < (n - 1) * 3:
            side2 = 2
            p2_i = n - 1
            p2_j = (n-1) - (p2 % (n - 1))
        else:
            side2 = 3
            p2_i = (n-1) - (p2 % (n - 1))
            p2_j = 0

        piece1 = np.roll(solution[p1_i,p1_j,:], side2 - side1)
        piece2 = np.roll(solution[p2_i,p2_j,:], side1 - side2)
        solution[p2_i,p2_j,:] = piece1
        solution[p1_i,p1_j,:] = piece2

        score += delta_matrix[p1,p2]
        if score < best_score:
            best_sol = solution.copy()
            best_score = score
        if score == 0:
            break
        pieces_to_update = {p1, p2}
        if p2 == 0:
            pieces_to_update.update([(n - 1) * 4 -1,1])
        else:
            pieces_to_update.update([p2-1,p2+1])
        if p1 == (n-1)*4-1:
            pieces_to_update.update([(n - 1) * 4 - 2, 0])
        else:
            pieces_to_update.update([p1 - 1, p1 + 1])

        for p1 in pieces_to_update:
            for p2 in range((n - 1) * 4):
                if p1 == p2:
                    continue

                if p1 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3] or p2 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3]:
                    if not (p1 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3] and p2 in [0, (n - 1), (n - 1) * 2, (n - 1) * 3]):
                        continue
                    else:
                        if p1 < p2:
                            delta_matrix[p2, p1] = eval_delta_corners(solution, p2, p1)
                        else:
                            delta_matrix[p1, p2] = eval_delta_corners(solution, p1, p2)
                else:
                    if p1 < p2:
                        delta_matrix[p2, p1] = eval_delta_edges(solution, p2, p1)
                    else:
                        delta_matrix[p1, p2] = eval_delta_edges(solution, p1, p2)

    return best_sol, nb_iter


def eval_border(solution):
    n = len(solution)
    top = solution[0,:n-1,1] != solution[0,1:,3]
    bot = solution[n-1,:n-1,1] != solution[n-1,1:,3]
    right = solution[:n-1,n-1,2] != solution[1:,n-1,0]
    left = solution[:n-1,0,2] != solution[1:,0,0]
    score = top.sum() + bot.sum() + right.sum() + left.sum()
    return score


def eval_delta_edges(solution, p1, p2):
    n = len(solution)
    if p1 < n - 1:
        side1 = 0
        p1_i = 0
        p1_j = p1
    elif p1 < (n - 1) * 2:
        side1 = 1
        p1_i = p1 % (n - 1)
        p1_j = n - 1
    elif p1 < (n - 1) * 3:
        side1 = 2
        p1_i = n - 1
        p1_j = (n - 1) - (p1 % (n - 1))
    else:
        side1 = 3
        p1_i = (n - 1) - (p1 % (n - 1))
        p1_j = 0
    if p2 < n - 1:
        side2 = 0
        p2_i = 0
        p2_j = p2
    elif p2 < (n - 1) * 2:
        side2 = 1
        p2_i = p2 % (n - 1)
        p2_j = n - 1
    elif p2 < (n - 1) * 3:
        side2 = 2
        p2_i = n - 1
        p2_j = (n - 1) - (p2 % (n - 1))
    else:
        side2 = 3
        p2_i = (n - 1) - (p2 % (n - 1))
        p2_j = 0
    if p2 == p1-1:
        if side1 == 0:
            old = (solution[p2_i, p2_j-1, 1] != solution[p2_i, p2_j, 3]).sum() + (solution[p2_i, p2_j, 1] != solution[p1_i, p1_j, 3]).sum() + (solution[p1_i, p1_j+1, 3] != solution[p1_i, p1_j, 1]).sum()
            new = (solution[p2_i, p2_j-1, 1] != solution[p1_i, p1_j, 3]).sum() + (solution[p2_i, p2_j, 3] != solution[p1_i, p1_j, 1]).sum() + (solution[p1_i, p1_j+1, 3] != solution[p2_i, p2_j, 1]).sum()
            return new - old
        elif side1 == 1:
            old = (solution[p2_i-1, p2_j, 2] != solution[p2_i, p2_j, 0]).sum() + (solution[p2_i, p2_j, 2] != solution[p1_i, p1_j, 0]).sum() + (solution[p1_i+1, p1_j, 0] != solution[p1_i, p1_j, 2]).sum()
            new = (solution[p2_i-1, p2_j, 2] != solution[p1_i, p1_j, 0]).sum() + (solution[p2_i, p2_j, 0] != solution[p1_i, p1_j, 2]).sum() + (solution[p1_i+1, p1_j, 0] != solution[p2_i, p2_j, 2]).sum()
            return new - old
        elif side1 == 2:
            old = (solution[p2_i, p2_j + 1, 3] != solution[p2_i, p2_j, 1]).sum() + (solution[p2_i, p2_j, 3] != solution[p1_i, p1_j, 1]).sum() + (solution[p1_i, p1_j - 1, 1] != solution[p1_i, p1_j, 3]).sum()
            new = (solution[p2_i, p2_j + 1, 3] != solution[p1_i, p1_j, 1]).sum() + (solution[p2_i, p2_j, 1] != solution[p1_i, p1_j, 3]).sum() + (solution[p1_i, p1_j - 1, 1] != solution[p2_i, p2_j, 3]).sum()
            return new - old
        else:
            old = (solution[p2_i+1, p2_j, 0] != solution[p2_i, p2_j, 2]).sum() + (solution[p2_i, p2_j, 0] != solution[p1_i, p1_j, 2]).sum() + (solution[p1_i-1, p1_j, 2] != solution[p1_i, p1_j, 0]).sum()
            new = (solution[p2_i+1, p2_j, 0] != solution[p1_i, p1_j, 2]).sum() + (solution[p2_i, p2_j, 2] != solution[p1_i, p1_j, 0]).sum() + (solution[p1_i-1, p1_j, 2] != solution[p2_i, p2_j, 0]).sum()
            return new - old
    else:
        if side1 == 0:
            color1 = solution[p1_i, p1_j-1, 1]
            color2 = solution[p1_i, p1_j+1, 3]
            color1_p1 = solution[p1_i, p1_j, 3]
            color2_p1 = solution[p1_i, p1_j, 1]
        elif side1 == 1:
            color1 = solution[p1_i-1, p1_j, 2]
            color2 = solution[p1_i+1, p1_j, 0]
            color1_p1 = solution[p1_i, p1_j, 0]
            color2_p1 = solution[p1_i, p1_j, 2]
        elif side1 == 2:
            color1 = solution[p1_i, p1_j + 1, 3]
            color2 = solution[p1_i, p1_j - 1, 1]
            color1_p1 = solution[p1_i, p1_j, 1]
            color2_p1 = solution[p1_i, p1_j, 3]
        else:
            color1 = solution[p1_i+1, p1_j, 0]
            color2 = solution[p1_i-1, p1_j, 2]
            color1_p1 = solution[p1_i, p1_j, 2]
            color2_p1 = solution[p1_i, p1_j, 0]
        if side2 == 0:
            color3 = solution[p2_i, p2_j - 1, 1]
            color4 = solution[p2_i, p2_j + 1, 3]
            color1_p2 = solution[p2_i, p2_j, 3]
            color2_p2 = solution[p2_i, p2_j, 1]
        elif side2 == 1:
            color3 = solution[p2_i - 1, p2_j, 2]
            color4 = solution[p2_i + 1, p2_j, 0]
            color1_p2 = solution[p2_i, p2_j, 0]
            color2_p2 = solution[p2_i, p2_j, 2]
        elif side2 == 2:
            color3 = solution[p2_i, p2_j + 1, 3]
            color4 = solution[p2_i, p2_j - 1, 1]
            color1_p2 = solution[p2_i, p2_j, 1]
            color2_p2 = solution[p2_i, p2_j, 3]
        else:
            color3 = solution[p2_i + 1, p2_j, 0]
            color4 = solution[p2_i - 1, p2_j, 2]
            color1_p2 = solution[p2_i, p2_j, 2]
            color2_p2 = solution[p2_i, p2_j, 0]

    old = (color1 != color1_p1).sum() + (color2 != color2_p1).sum() + (color3 != color1_p2).sum() + (color4 != color2_p2).sum()
    new = (color1 != color1_p2).sum() + (color2 != color2_p2).sum() + (color3 != color1_p1).sum() + (color4 != color2_p1).sum()
    return new - old


def eval_delta_corners(solution, p1, p2):
    n = len(solution)
    if p1 < n - 1:
        side1 = 0
        p1_i = 0
        p1_j = p1
    elif p1 < (n - 1) * 2:
        side1 = 1
        p1_i = p1 % (n - 1)
        p1_j = n - 1
    elif p1 < (n - 1) * 3:
        side1 = 2
        p1_i = n - 1
        p1_j = (n - 1) - (p1 % (n - 1))
    else:
        side1 = 3
        p1_i = (n - 1) - (p1 % (n - 1))
        p1_j = 0
    if p2 < n - 1:
        side2 = 0
        p2_i = 0
        p2_j = p2
    elif p2 < (n - 1) * 2:
        side2 = 1
        p2_i = p2 % (n - 1)
        p2_j = n - 1
    elif p2 < (n - 1) * 3:
        side2 = 2
        p2_i = n - 1
        p2_j = (n - 1) - (p2 % (n - 1))
    else:
        side2 = 3
        p2_i = (n - 1) - (p2 % (n - 1))
        p2_j = 0
    if side1 == 0:
        color1 = solution[p1_i+1, p1_j, 0]
        color2 = solution[p1_i, p1_j + 1, 3]
        color1_p1 = solution[p1_i, p1_j, 2]
        color2_p1 = solution[p1_i, p1_j, 1]
    elif side1 == 1:
        color1 = solution[p1_i, p1_j-1, 1]
        color2 = solution[p1_i + 1, p1_j, 0]
        color1_p1 = solution[p1_i, p1_j, 3]
        color2_p1 = solution[p1_i, p1_j, 2]
    elif side1 == 2:
        color1 = solution[p1_i-1, p1_j, 2]
        color2 = solution[p1_i, p1_j - 1, 1]
        color1_p1 = solution[p1_i, p1_j, 0]
        color2_p1 = solution[p1_i, p1_j, 3]
    else:
        color1 = solution[p1_i, p1_j+1, 3]
        color2 = solution[p1_i - 1, p1_j, 2]
        color1_p1 = solution[p1_i, p1_j, 1]
        color2_p1 = solution[p1_i, p1_j, 0]
    if side2 == 0:
        color3 = solution[p2_i+1, p2_j, 0]
        color4 = solution[p2_i, p2_j + 1, 3]
        color1_p2 = solution[p2_i, p2_j, 2]
        color2_p2 = solution[p2_i, p2_j, 1]
    elif side2 == 1:
        color3 = solution[p2_i, p2_j-1, 1]
        color4 = solution[p2_i + 1, p2_j, 0]
        color1_p2 = solution[p2_i, p2_j, 3]
        color2_p2 = solution[p2_i, p2_j, 2]
    elif side2 == 2:
        color3 = solution[p2_i-1, p2_j, 2]
        color4 = solution[p2_i, p2_j - 1, 1]
        color1_p2 = solution[p2_i, p2_j, 0]
        color2_p2 = solution[p2_i, p2_j, 3]
    else:
        color3 = solution[p2_i, p2_j+1, 3]
        color4 = solution[p2_i - 1, p2_j, 2]
        color1_p2 = solution[p2_i, p2_j, 1]
        color2_p2 = solution[p2_i, p2_j, 0]
    return ((color1 != color1_p2).sum() + (color2 != color2_p2).sum() + (color3 != color1_p1).sum() + (color4 != color2_p1)).sum() - ((color1 != color1_p1).sum() + (color2 != color2_p1).sum() + (color3 != color1_p2).sum() + (color4 != color2_p2)).sum()

########################################################################################################################
########################################################################################################################
###########                                  Traitement de l'intérieur                                      ############
########################################################################################################################
########################################################################################################################


def solve_inside(solution, tabu_length, temp_best_score, patience_for_SA, max_duration, start_time):
    n = len(solution)
    n_pieces = (n - 2) ** 2

    score = evaluation(solution)
    best_sol = solution.copy()
    best_score = score
    nb_iter = 0

    tabu_dict = dict()

    delta_matrix = create_delta_matrix(solution)

    stagnating = 0
    while stagnating < patience_for_SA and time.time() < max_duration + start_time and best_score > 0:
        nb_iter += 1

        best_moves = []
        best_cost = 10
        for p1 in range(n_pieces):
            for p2 in range(p1):
                for roll1 in range(4):
                    for roll2 in range(4):
                        temp_cost = delta_matrix[p1, p2, roll1, roll2]
                        if score + temp_cost < temp_best_score:
                            if temp_cost < best_cost:
                                best_cost = temp_cost
                                best_moves = [(p1, p2, roll1, roll2)]
                            elif temp_cost == best_cost:
                                best_moves.append((p1, p2, roll1, roll2))
                        elif (p1, p2) not in tabu_dict or tabu_dict[(p1, p2)] + tabu_length < nb_iter:
                            if temp_cost < best_cost:
                                best_cost = temp_cost
                                best_moves = [(p1, p2, roll1, roll2)]
                            elif temp_cost == best_cost:
                                best_moves.append((p1, p2, roll1, roll2))

        p1, p2, roll1, roll2 = best_moves[np.random.randint(len(best_moves))]
        tabu_dict[(p1, p2)] = nb_iter

        p1_i = (p1 // (n-2)) + 1
        p1_j = (p1 % (n-2)) + 1
        p2_i = (p2 // (n-2)) + 1
        p2_j = (p2 % (n-2)) + 1

        piece1 = np.roll(solution[p1_i, p1_j, :], roll1)
        piece2 = np.roll(solution[p2_i, p2_j, :], roll2)
        solution[p2_i, p2_j, :] = piece1
        solution[p1_i, p1_j, :] = piece2

        delta = delta_matrix[p1, p2, roll1, roll2]
        score += delta

        if score < temp_best_score:
            temp_best_score = score
        if score < best_score:
            stagnating = 0
            best_sol = solution.copy()
            best_score = score
        else:
            stagnating += 1
        if score == 0:
            break

        pieces_to_update = {p1,p2}
        if p1_i > 1:
            pieces_to_update.add(p1-n+2)
        if p1_i < n-2:
            pieces_to_update.add(p1 + n - 2)
        if p1_j > 1 :
            pieces_to_update.add(p1 - 1)
        if p1_j < n-2:
            pieces_to_update.add(p1 + 1)
        if p2_i > 1:
            pieces_to_update.add(p2-n+2)
        if p2_i < n-2:
            pieces_to_update.add(p2 + n - 2)
        if p2_j > 1 :
            pieces_to_update.add(p2 - 1)
        if p2_j < n-2:
            pieces_to_update.add(p2 + 1)

        for p1 in pieces_to_update:
            for p2 in range(n_pieces):
                if p1 == p2:
                    continue
                for roll1 in range(4):
                    for roll2 in range(4):
                        if p1 < p2:
                            delta_matrix[p2, p1, roll2, roll1] = eval_delta_inside(solution, p2, p1, roll2, roll1)
                        else:
                            delta_matrix[p1, p2, roll1, roll2] = eval_delta_inside(solution, p1, p2, roll1, roll2)

    return best_sol, best_score, nb_iter


def eval_delta_inside(solution, p1, p2, roll1, roll2):
    n = len(solution)
    p1_i = (p1 // (n-2)) + 1
    p1_j = (p1 % (n-2)) + 1
    p2_i = (p2 // (n-2)) + 1
    p2_j = (p2 % (n-2)) + 1
    piece1_old = solution[p1_i,p1_j,:]
    piece1_new = np.roll(piece1_old, roll1)
    piece2_old = solution[p2_i, p2_j, :]
    piece2_new = np.roll(piece2_old, roll2)
    if p2_i == p1_i and p2_j == p1_j - 1:
        surrounding1_old = np.array([solution[p1_i-1, p1_j, 2], solution[p1_i, p1_j+1, 3], solution[p1_i+1, p1_j, 0], solution[p1_i, p1_j-1, 1]], dtype=int)
        surrounding2_old = np.array([solution[p2_i - 1, p2_j, 2], solution[p2_i, p2_j + 1, 3], solution[p2_i + 1, p2_j, 0],solution[p2_i, p2_j - 1, 1]], dtype=int)
        surrounding1_new = np.array([solution[p1_i - 1, p1_j, 2], solution[p1_i, p1_j + 1, 3], solution[p1_i + 1, p1_j, 0],piece1_new[1]], dtype=int)
        surrounding2_new = np.array([solution[p2_i - 1, p2_j, 2], piece2_new[3], solution[p2_i + 1, p2_j, 0],solution[p2_i, p2_j - 1, 1]], dtype=int)
        old = (surrounding1_old != piece1_old).sum() + (surrounding2_old != piece2_old).sum() - (piece1_old[3] != piece2_old[1]).sum()
        new = (surrounding2_new != piece1_new).sum() + (surrounding1_new != piece2_new).sum() - (piece1_new[1] != piece2_new[3]).sum()
        return new - old
    elif p1 - n + 2 == p2:
        surrounding1_old = np.array([solution[p1_i - 1, p1_j, 2], solution[p1_i, p1_j + 1, 3], solution[p1_i + 1, p1_j, 0], solution[p1_i, p1_j - 1, 1]], dtype=int)
        surrounding2_old = np.array([solution[p2_i - 1, p2_j, 2], solution[p2_i, p2_j + 1, 3], solution[p2_i + 1, p2_j, 0], solution[p2_i, p2_j - 1, 1]], dtype=int)
        surrounding1_new = np.array([piece1_new[2], solution[p1_i, p1_j + 1, 3], solution[p1_i + 1, p1_j, 0], solution[p1_i, p1_j - 1, 1]], dtype=int)
        surrounding2_new = np.array([solution[p2_i - 1, p2_j, 2], solution[p2_i, p2_j + 1, 3], piece2_new[0], solution[p2_i, p2_j - 1, 1]], dtype=int)
        old = (surrounding1_old != piece1_old).sum() + (surrounding2_old != piece2_old).sum() - (piece1_old[0] != piece2_old[2]).sum()
        new = (surrounding2_new != piece1_new).sum() + (surrounding1_new != piece2_new).sum() - (piece1_new[2] != piece2_new[0]).sum()
        return new - old
    else:
        surrounding1 = np.array([solution[p1_i - 1, p1_j, 2], solution[p1_i, p1_j + 1, 3], solution[p1_i + 1, p1_j, 0], solution[p1_i, p1_j - 1, 1]], dtype=int)
        surrounding2 = np.array([solution[p2_i - 1, p2_j, 2], solution[p2_i, p2_j + 1, 3], solution[p2_i + 1, p2_j, 0], solution[p2_i, p2_j - 1, 1]], dtype=int)

        new = (surrounding2 != piece1_new).sum() + (surrounding1 != piece2_new).sum()
        old = (surrounding1 != piece1_old).sum() + (surrounding2 != piece2_old).sum()
        return new - old


def create_delta_matrix(solution):
    n = len(solution)
    n_pieces = (n - 2) ** 2
    delta_matrix = np.zeros((n_pieces,n_pieces,4,4), dtype=np.int8)
    for p1 in range(n_pieces):
        for p2 in range(p1):
            for roll1 in range(4):
                for roll2 in range(4):
                    delta_matrix[p1, p2, roll1, roll2] = eval_delta_inside(solution, p1, p2, roll1, roll2)
    return delta_matrix


def simulated_annealing(solution, t0, cooling, patience_for_ILS, max_duration, start_time):
    n = len(solution)
    n_pieces = (n-2)**2
    temperature = t0
    nb_iter = 0
    stagnating = 0
    score = evaluation(solution)

    delta_matrix = create_delta_matrix(solution)

    while stagnating < patience_for_ILS and time.time() < max_duration + start_time:
        nb_iter += 1

        p1 = np.random.randint(1,n_pieces)
        p2 = np.random.randint(p1)
        roll1 = np.random.randint(4)
        roll2 = np.random.randint(4)

        delta = delta_matrix[p1, p2, roll1, roll2]
        proba = np.exp(-delta / temperature)
        temperature *= cooling

        if delta < 0:
            return solution, score, True, nb_iter
        elif np.random.rand() > proba:
            stagnating += 1
            continue

        score += delta
        stagnating += 1

        p1_i = (p1 // (n-2)) + 1
        p1_j = (p1 % (n-2)) + 1
        p2_i = (p2 // (n-2)) + 1
        p2_j = (p2 % (n-2)) + 1

        piece1 = np.roll(solution[p1_i, p1_j, :], roll1)
        piece2 = np.roll(solution[p2_i, p2_j, :], roll2)
        solution[p2_i, p2_j, :] = piece1
        solution[p1_i, p1_j, :] = piece2

        pieces_to_update = {p1, p2}
        if p1_i > 1:
            pieces_to_update.add(p1 - n + 2)
        if p1_i < n - 2:
            pieces_to_update.add(p1 + n - 2)
        if p1_j > 1:
            pieces_to_update.add(p1 - 1)
        if p1_j < n - 2:
            pieces_to_update.add(p1 + 1)
        if p2_i > 1:
            pieces_to_update.add(p2 - n + 2)
        if p2_i < n - 2:
            pieces_to_update.add(p2 + n - 2)
        if p2_j > 1:
            pieces_to_update.add(p2 - 1)
        if p2_j < n - 2:
            pieces_to_update.add(p2 + 1)

        for p1 in pieces_to_update:
            for p2 in range(n_pieces):
                if p1 == p2:
                    continue
                for roll1 in range(4):
                    for roll2 in range(4):
                        if p1 < p2:
                            delta_matrix[p2, p1, roll2, roll1] = eval_delta_inside(solution, p2, p1, roll2, roll1)
                        else:
                            delta_matrix[p1, p2, roll1, roll2] = eval_delta_inside(solution, p1, p2, roll1, roll2)
    return solution, score, False, nb_iter


def perturbation(solution, perturbation_ratio):
    n = len(solution)
    n_pieces = n**2
    nb_of_swap = int(n_pieces * perturbation_ratio)
    corners = [(0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 0)]
    edges = [(0,k) for k in range(1,n-1)] + [(k,n-1) for k in range(1,n-1)] + [(n-1,k) for k in range(n-2,0,-1)] + [(k,0) for k in range(n-2,0,-1)]
    for i in range(nb_of_swap):
        type_of_piece = np.random.rand()
        if type_of_piece < 4/n_pieces:
            p1, p2 = np.random.choice(4,2,False)
            p1_i, p1_j = corners[p1]
            p2_i, p2_j = corners[p2]
            piece1 = np.roll(solution[p1_i, p1_j,:], p2-p1)
            piece2 = np.roll(solution[p2_i, p2_j,:], p1-p2)
            solution[p1_i, p1_j, :] = piece2
            solution[p2_i, p2_j, :] = piece1
        elif type_of_piece < (n-1)*4/n_pieces:
            p1, p2 = np.random.choice((n - 2) * 4, 2, False)
            p1_i, p1_j = edges[p1]
            p2_i, p2_j = edges[p2]
            side1 = p1 // (n-2)
            side2 = p2 // (n-2)
            piece1 = np.roll(solution[p1_i, p1_j, :], side2 - side1)
            piece2 = np.roll(solution[p2_i, p2_j, :], side1 - side2)
            solution[p1_i, p1_j, :] = piece2
            solution[p2_i, p2_j, :] = piece1
        else:
            p1, p2 = np.random.choice((n-2)**2, 2, False)
            p1_i, p1_j = p1//((n-2)) + 1, p1%((n-2)) + 1
            p2_i, p2_j = p2//((n-2)) + 1, p2%((n-2)) + 1
            piece1 = np.roll(solution[p1_i, p1_j, :], np.random.randint(4))
            piece2 = np.roll(solution[p2_i, p2_j, :], np.random.randint(4))
            solution[p1_i, p1_j, :] = piece2
            solution[p2_i, p2_j, :] = piece1
    return solution
