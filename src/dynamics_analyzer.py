"""Classe DynamicsAnalyzer : analyse qualitative complète d'un système dynamique (linéaire ou non-linéaire)."""
import numpy as np
from typing import Callable, Tuple, Union, Dict

class DynamicsAnalyzer:
    def generate_report(self, filename="rapport_analyse.pdf"):
        """
        Génère un PDF pédagogique complet avec :
        1. Système étudié
        2. Valeurs propres et classification
        3. Vecteurs propres et sous-espaces
        4. Graphiques d'isoclines orientées
        5. Analyse par quadrant
        6. Portrait de phase final
        7. Conclusion sur le comportement dynamique
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import io
        def add_text_page(pdf, title, content):
            fig, ax = plt.subplots(figsize=(8.3, 11.7))
            ax.axis('off')
            ax.text(0.5, 0.95, title, fontsize=18, ha='center', va='top', weight='bold')
            ax.text(0.05, 0.9, content, fontsize=12, ha='left', va='top', wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
        with PdfPages(filename) as pdf:
            # 1. Système étudié
            sys_txt = "Système étudié :\n"
            if isinstance(self.system, np.ndarray):
                sys_txt += f"Système linéaire :\nA = {self.system}\n\nÉquations :\n"
                sys_txt += f"x' = {self.system[0,0]:.2f} x + {self.system[0,1]:.2f} y\n"
                sys_txt += f"y' = {self.system[1,0]:.2f} x + {self.system[1,1]:.2f} y"
            else:
                sys_txt += "Système non-linéaire :\n"
                sys_txt += "x' = f₁(x, y)\ny' = f₂(x, y)\n"
            sys_txt += f"\nDomaine d'étude : {self.domain}"
            add_text_page(pdf, "1. Système étudié", sys_txt + "\n\nCe système sera analysé qualitativement selon la méthode en 6 étapes.")

            # 2. Valeurs propres et classification
            try:
                eig = self.analyze_eigenvalues()
                eig_txt = f"Valeurs propres : {eig['valeurs_propres']}\nClassification : {eig['classification']}\nNature : {eig['nature']}\n\n"
                eig_txt += "\nLes valeurs propres déterminent la stabilité locale du point fixe.\n"
            except Exception as e:
                eig_txt = f"Non applicable (non-linéaire ou erreur) : {e}"
            add_text_page(pdf, "2. Valeurs propres et classification", eig_txt)

            # 3. Vecteurs propres et sous-espaces
            try:
                esp = self.compute_eigenspaces(plot=False)
                esp_txt = f"Eₛ (stable) : dim={esp['E_s_dim']} base={esp['E_s_basis']}\n"
                esp_txt += f"Eᵤ (instable) : dim={esp['E_u_dim']} base={esp['E_u_basis']}\n"
                esp_txt += f"E꜀ (centre) : dim={esp['E_c_dim']} base={esp['E_c_basis']}\n\n"
                esp_txt += "\nLes sous-espaces propres structurent le flux autour du point fixe.\n"
            except Exception as e:
                esp_txt = f"Non applicable (non-linéaire ou erreur) : {e}"
            add_text_page(pdf, "3. Vecteurs propres et sous-espaces", esp_txt)

            # 4. Graphique isoclines
            self.plot_isoclines(plot=False)
            pdf.savefig(plt.gcf())
            plt.close()
            add_text_page(pdf, "4. Isoclines orientées", "Les isoclines ẋ=0 et ẏ=0 séparent les régions de directions opposées. Les flèches indiquent l'orientation du mouvement sur chaque isocline.")

            # 5. Analyse par quadrant
            try:
                quad = self.quadrant_analysis()
                quad_txt = "Quadrant | Point | Signe (ẋ, ẏ) | Sens\n"
                for q in quad:
                    quad_txt += f"{q['quadrant']} | {q['point']} | {q['sign']} | {q['sens']}\n"
                quad_txt += "\nChaque quadrant indique le sens général du flux local.\n"
            except Exception as e:
                quad_txt = f"Non applicable : {e}"
            add_text_page(pdf, "5. Analyse par quadrant", quad_txt)

            # 6. Portrait de phase final
            self.plot_final_phase_portrait(plot=False)
            pdf.savefig(plt.gcf())
            plt.close()
            add_text_page(pdf, "6. Portrait de phase final", "Ce graphique intègre toutes les informations : isoclines, directions principales, sous-espaces, trajectoires, points fixes et flux par quadrant.")

            # 7. Conclusion
            concl = "Le comportement dynamique global dépend de la nature des valeurs propres et des sous-espaces.\n"
            if 'eigen' in self.report:
                nat = self.report['eigen']['nature']
                if 'nœud' in nat.lower() or 'stable' in nat.lower():
                    concl += "Le système converge vers le point fixe (stable)."
                elif 'instable' in nat.lower():
                    concl += "Le système diverge du point fixe (instable)."
                elif 'selle' in nat.lower():
                    concl += "Le point fixe est une selle : certaines directions sont stables, d'autres instables."
                elif 'centre' in nat.lower():
                    concl += "Le système présente des oscillations périodiques (centre)."
                else:
                    concl += f"Nature : {nat}"
            else:
                concl += "Voir le portrait de phase pour l'interprétation qualitative."
            add_text_page(pdf, "7. Conclusion", concl)
        return filename
    def plot_final_phase_portrait(self, n_trajectories=8, traj_len=10, plot=True):
        """
        Portrait de phase de haut niveau avec style professionnel :
        - Champ de vecteurs en arrière-plan
        - Trajectoires colorées avec conditions initiales
        - Variétés stables/instables mises en évidence
        - Formatage LaTeX pour les équations
        """
        import matplotlib.pyplot as plt
        from scipy.integrate import solve_ivp
        
        # Définition du système
        if isinstance(self.system, np.ndarray):
            A = self.system
            def system_ode(t, y):
                return A @ y
            def field_vector(x, y):
                return A[0,0]*x + A[0,1]*y, A[1,0]*x + A[1,1]*y
            # Classification pour le titre
            try:
                eigvals = np.linalg.eigvals(A)
                if 'eigen' in self.report:
                    nature = self.report['eigen']['nature']
                else:
                    nature = "Point fixe"
                # Équations pour le titre
                title_eq = f"$\\dot{{x}} = {A[0,0]:.2f}x + {A[0,1]:.2f}y, \\quad \\dot{{y}} = {A[1,0]:.2f}x + {A[1,1]:.2f}y$"
            except:
                nature = "Système linéaire"
                title_eq = "$\\dot{\\mathbf{x}} = A\\mathbf{x}$"
        else:
            def system_ode(t, y):
                return [self.system[0](y[0], y[1]), self.system[1](y[0], y[1])]
            def field_vector(x, y):
                return self.system[0](x, y), self.system[1](x, y)
            nature = "Système non-linéaire"
            title_eq = "$\\dot{x} = f_1(x,y), \\quad \\dot{y} = f_2(x,y)$"
        
        # Paramètres du tracé
        x_min, x_max, y_min, y_max = self.domain
        num_points = 15
        t_span = [0, traj_len]
        t_eval = np.linspace(t_span[0], t_span[1], 500)
        
        # Création de la grille pour le champ de vecteurs
        x_grid = np.linspace(x_min, x_max, num_points)
        y_grid = np.linspace(y_min, y_max, num_points)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Calcul du champ de vecteurs
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                u_val, v_val = field_vector(X[i, j], Y[i, j])
                U[i, j] = u_val
                V[i, j] = v_val
        
        # Conditions initiales variées
        if isinstance(self.system, np.ndarray):
            # Pour systèmes linéaires : conditions autour de l'origine
            conditions_initiales = [
                [2.5, 0.0], [-2.5, 0.0],  # Sur l'axe x
                [0.0, 2.5], [0.0, -2.5],  # Sur l'axe y
                [1.8, 1.8], [-1.8, 1.8],  # Quadrants I et II
                [-1.8, -1.8], [1.8, -1.8] # Quadrants III et IV
            ]
        else:
            # Pour systèmes non-linéaires : conditions réparties
            conditions_initiales = [
                [2.0, 0.0], [-2.0, 0.0],
                [0.0, 2.0], [0.0, -2.0],
                [1.5, 1.5], [-1.5, 1.5],
                [-1.5, -1.5], [1.5, -1.5]
            ]
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 1. Champ de vecteurs en arrière-plan
        ax.quiver(X, Y, U, V, scale=30, color='lightblue', alpha=0.6, width=0.005)
        
        # 2. Trajectoires avec couleurs distinctes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, y0 in enumerate(conditions_initiales):
            try:
                sol = solve_ivp(system_ode, t_span, y0, t_eval=t_eval, method='RK45')
                ax.plot(sol.y[0], sol.y[1], color=colors[i], linewidth=2, 
                       label=f'$y_0=({y0[0]}, {y0[1]})$')
                # Point initial
                ax.plot(sol.y[0][0], sol.y[1][0], 'o', color=colors[i], markersize=6)
            except Exception as e:
                continue
        
        # 3. Variétés stables/instables (si applicable)
        if isinstance(self.system, np.ndarray) and 'eigenspaces' in self.report:
            esp_data = self.report['eigenspaces']
            
            # Variété stable (E_s)
            if esp_data['E_s_basis']:
                for v in esp_data['E_s_basis']:
                    # Tracer la ligne dans la direction du vecteur propre stable
                    scale = max(x_max - x_min, y_max - y_min) / 2
                    ax.plot([-scale*v[0], scale*v[0]], [-scale*v[1], scale*v[1]], 
                           'darkblue', linestyle='-', linewidth=4, 
                           label='Variété stable $W^s$')
            
            # Variété instable (E_u)
            if esp_data['E_u_basis']:
                for v in esp_data['E_u_basis']:
                    scale = max(x_max - x_min, y_max - y_min) / 2
                    ax.plot([-scale*v[0], scale*v[0]], [-scale*v[1], scale*v[1]], 
                           'darkred', linestyle='-', linewidth=4, 
                           label='Variété instable $W^u$')
        
        # 4. Axes et grille
        ax.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # 5. Mise en forme
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_title(f'Portrait de phase : {title_eq}\n({nature})', fontsize=16, pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.axis('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        if plot:
            plt.show()
        else:
            # Ne ferme pas la figure pour Streamlit
            pass
    def __init__(self, system: Union[np.ndarray, Tuple[Callable, Callable]], domain: Tuple[float, float, float, float]=(-5,5,-5,5)):
        self.system = system
        self.domain = domain
        self.report = {}

    def analyze_eigenvalues(self) -> Dict:
        if not isinstance(self.system, np.ndarray):
            raise ValueError("L'analyse des valeurs propres s'applique uniquement aux systèmes linéaires (matrice A)")
        A = self.system
        eigvals, eigvecs = np.linalg.eig(A)
        classification = []
        nature = []
        for val in eigvals:
            if np.isclose(val.imag, 0):
                if val.real < 0:
                    cls = 'Stable'
                elif val.real > 0:
                    cls = 'Instable'
                else:
                    cls = 'Centre'
            else:
                if val.real < 0:
                    cls = 'Stable (spirale)'
                elif val.real > 0:
                    cls = 'Instable (spirale)'
                else:
                    cls = 'Centre (cycle limite)'
            classification.append(cls)
        # Nature du point fixe
        if np.all(np.isreal(eigvals)):
            if np.all(eigvals < 0):
                nat = 'Nœud stable'
            elif np.all(eigvals > 0):
                nat = 'Nœud instable'
            elif np.any(eigvals < 0) and np.any(eigvals > 0):
                nat = 'Selle'
            else:
                nat = 'Cas dégénéré'
        else:
            if np.all(np.real(eigvals) < 0):
                nat = 'Foyer stable'
            elif np.all(np.real(eigvals) > 0):
                nat = 'Foyer instable'
            elif np.all(np.real(eigvals) == 0):
                nat = 'Centre'
            else:
                nat = 'Foyer-selle'
        # Rapport
        report = {
            'valeurs_propres': eigvals,
            'vecteurs_propres': eigvecs,
            'classification': classification,
            'nature': nat
        }
        self.report['eigen'] = report
        return report


    def compute_eigenspaces(self, plot=True):
        """
        Calcule vecteurs propres, sous-espaces stables/instables/centres, dimensions, bases, et trace les directions principales.
        """
        import matplotlib.pyplot as plt
        if not isinstance(self.system, np.ndarray):
            raise ValueError("Eigenspaces: système linéaire requis (matrice A)")
        A = self.system
        eigvals, eigvecs = np.linalg.eig(A)
        E_s, E_u, E_c = [], [], []
        for i, val in enumerate(eigvals):
            v = eigvecs[:, i].real
            if np.isclose(val.imag, 0):
                if val.real < 0:
                    E_s.append(v)
                elif val.real > 0:
                    E_u.append(v)
                else:
                    E_c.append(v)
            else:
                if np.isclose(val.real, 0):
                    E_c.append(v)
                elif val.real < 0:
                    E_s.append(v)
                else:
                    E_u.append(v)
        report = {
            'E_s_dim': len(E_s), 'E_s_basis': E_s,
            'E_u_dim': len(E_u), 'E_u_basis': E_u,
            'E_c_dim': len(E_c), 'E_c_basis': E_c
        }
        self.report['eigenspaces'] = report
        if plot:
            plt.figure(figsize=(5,5))
            plt.axhline(0, color='k', lw=0.5)
            plt.axvline(0, color='k', lw=0.5)
            colors = {'E_s':'b', 'E_u':'r', 'E_c':'g'}
            for v in E_s:
                plt.arrow(0,0,v[0],v[1],color=colors['E_s'],width=0.05,head_width=0.2,length_includes_head=True,label='Stable')
            for v in E_u:
                plt.arrow(0,0,v[0],v[1],color=colors['E_u'],width=0.05,head_width=0.2,length_includes_head=True,label='Instable')
            for v in E_c:
                plt.arrow(0,0,v[0],v[1],color=colors['E_c'],width=0.05,head_width=0.2,length_includes_head=True,label='Centre')
            plt.xlim(-3,3)
            plt.ylim(-3,3)
            plt.title('Directions principales (espaces propres)')
            plt.grid(True)
            plt.show()
        return report

    def plot_isoclines(self, plot=True):
        """
        Trace les isoclines ẋ=0 et ẏ=0, colorie les régions, ajoute flèches, marque intersections, carte des directions.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        if isinstance(self.system, np.ndarray):
            def f(x, y):
                A = self.system
                return A[0,0]*x + A[0,1]*y, A[1,0]*x + A[1,1]*y
        else:
            def f(x, y):
                return self.system[0](x, y), self.system[1](x, y)
        x_min, x_max, y_min, y_max = self.domain
        x = np.linspace(x_min, x_max, 200)
        y = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x, y)
        U, V = f(X, Y)
        fig, ax = plt.subplots(figsize=(6,6))
        # Isoclines (labels supprimés car non pris en charge par contour)
        ax.contour(X, Y, U, levels=[0], colors='b', linewidths=2, linestyles='--')
        ax.contour(X, Y, V, levels=[0], colors='r', linewidths=2, linestyles='-.')
        # Color regions
        cmap = ListedColormap(['#d0f0c0','#f0d0c0'])
        ax.contourf(X, Y, np.sign(U), alpha=0.1, cmap='Blues')
        ax.contourf(X, Y, np.sign(V), alpha=0.1, cmap='Reds')
        # Arrows on isoclines
        for c in [0]:
            iso_x = ax.contour(X, Y, U, levels=[c], colors='b', linewidths=2)
            iso_y = ax.contour(X, Y, V, levels=[c], colors='r', linewidths=2)
            # Compatibilité: certaines versions n'exposent pas 'collections'
            collections_x = getattr(iso_x, 'collections', [])
            collections_y = getattr(iso_y, 'collections', [])
            for collection in list(collections_x) + list(collections_y):
                # get_paths peut ne pas exister sur certains artistes; on vérifie
                get_paths = getattr(collection, 'get_paths', None)
                if not callable(get_paths):
                    continue
                for path in get_paths():
                    v = path.vertices
                    if len(v) > 10:
                        idx = len(v)//2
                        dx, dy = f(v[idx,0], v[idx,1])
                        ax.arrow(v[idx,0], v[idx,1], dx/10, dy/10, head_width=0.2, color='k')
        # Points fixes
        if isinstance(self.system, np.ndarray):
            try:
                from numpy.linalg import solve
                fixed = solve(self.system, np.zeros(2))
                ax.plot(fixed[0], fixed[1], 'ko', label='Point fixe')
            except Exception:
                pass
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Isoclines et directions')
        ax.grid(True)
        if plot:
            plt.show()
        else:
            # Ne ferme pas la figure pour Streamlit
            pass

    def quadrant_analysis(self):
        """
        Analyse les quadrants autour du point fixe, signe de (ẋ, ẏ), sens du flux, tableau récapitulatif, séparatrices.
        """
        if isinstance(self.system, np.ndarray):
            def f(x, y):
                A = self.system
                return A[0,0]*x + A[0,1]*y, A[1,0]*x + A[1,1]*y
            try:
                from numpy.linalg import solve
                fixed = solve(self.system, np.zeros(2))
            except Exception:
                fixed = np.zeros(2)
        else:
            def f(x, y):
                return self.system[0](x, y), self.system[1](x, y)
            fixed = np.zeros(2)
        x0, y0 = fixed[0], fixed[1]
        # Quadrants: (+,+), (-,+), (-,-), (+,-)
        dx = (self.domain[1] - self.domain[0]) / 4
        dy = (self.domain[3] - self.domain[2]) / 4
        quadrants = [
            (x0+dx, y0+dy),
            (x0-dx, y0+dy),
            (x0-dx, y0-dy),
            (x0+dx, y0-dy)
        ]
        summary = []
        for i, (xq, yq) in enumerate(quadrants):
            u, v = f(xq, yq)
            sign = (np.sign(u), np.sign(v))
            if sign == (0,0):
                sens = 'Stationnaire'
            elif sign[0] == 0:
                sens = 'Verticale'
            elif sign[1] == 0:
                sens = 'Horizontale'
            elif sign == (1,1):
                sens = 'Divergent'
            elif sign == (-1,-1):
                sens = 'Convergent'
            else:
                sens = 'Tournant'
            summary.append({'quadrant':i+1,'point':(xq,yq),'sign':sign,'sens':sens})
        self.report['quadrants'] = summary
        return summary

    def plot_phase_portrait(self):
        # ...à implémenter...
        pass

    def full_analysis(self):
        steps = []
        try:
            steps.append(self.analyze_eigenvalues())
        except Exception as e:
            steps.append(f"Erreur valeurs propres: {e}")
        # ...appeler les autres étapes...
        return steps
