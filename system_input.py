import numpy as np

class LinearSystemInput:
    def get_matrix_input(self):
        print("\nEntrez la matrice A (format: [[a, b], [c, d]]):")
        print("Exemple: [[1, 2], [3, 4]]")
        while True:
            try:
                matrix_str = input("A = ").strip()
                matrix = eval(matrix_str)
                matrix = np.array(matrix, dtype=float)
                if matrix.shape == (2, 2):
                    return matrix
                else:
                    print("La matrice doit être 2x2.")
            except:
                print("Format invalide. Réessayez.")
    def get_domain_input(self):
        print("\nDomaine d'étude [xmin, xmax, ymin, ymax]:")
        print("Exemple: -5 5 -5 5")
        while True:
            try:
                domain_str = input("Domaine: ").strip()
                domain = list(map(float, domain_str.split()))
                if len(domain) == 4:
                    return domain
                else:
                    print("Entrez 4 valeurs.")
            except:
                print("Format invalide. Réessayez.")

class NonlinearSystemInput:
    def get_functions_input(self):
        print("\nEntrez les fonctions f1(x,y) et f2(x,y):")
        print("Exemple: x*(1 - x - y)")
        functions = []
        functions.append(input("f1(x,y) = dx/dt = ").strip())
        functions.append(input("f2(x,y) = dy/dt = ").strip())
        return functions
    def get_domain_input(self):
        return LinearSystemInput().get_domain_input()
