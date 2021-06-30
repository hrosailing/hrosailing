import hrosailing.polardiagram as pol
import matplotlib.pyplot as plt


def print_polar_table():
    polar_diagram = pol.from_csv(
        'C:/Users/MSI/PycharmProjects/PolarDiagrams/src/polar_diagrams/download/Dehler34.txt',
        fmt='array', tw=True)
    print(polar_diagram)
    

def print_point_cloud():
    polar_diagram = pol.from_csv(
        'C:/Users/MSI/PycharmProjects/PolarDiagrams/src/data/test/from_Dehler34_rep10000_noise1',
        fmt='hro', tw=True)
    print(polar_diagram)


def symmetrize_table():
    not_sym = pol.from_csv(
        'C:/Users/MSI/PycharmProjects/PolarDiagrams/src/polar_diagrams/download/Dehler34.csv',
        fmt='hro', tw=True)
    sym = pol.symmetric_polar_diagram(not_sym)
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
    not_sym.polar_plot(ws_range=[6, 8, 10, 12, 14], ax=ax1, 
                       colors=('deeppink', 'cyan'),
                       show_legend=False, ls='-', lw=0.8, ms=0.3)
    sym.polar_plot(ws_range=[6, 8, 10, 12, 14], ax=ax2, 
                   colors=('deeppink', 'cyan'),
                   show_legend=False, ls='-', lw=0.8, ms=0.3)
    plt.show()


def plot_polar_table():
    polar_diagram = pol.from_csv(
        'C:/Users/MSI/PycharmProjects/PolarDiagrams/src/polar_diagrams/download/Dehler34.txt',
        fmt='array', tw=True)
    polar_diagram = pol.symmetric_polar_diagram(polar_diagram)
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.667, 0.2, 0.2), polar=True)
    ax2 = fig.add_axes((0.334, 0.667, 0.2, 0.2))
    ax3 = fig.add_axes((0.6, 0.667, 0.2, 0.2), projection='3d')
    ax4 = fig.add_axes((0.1, 0.25, 0.2, 0.2))
    ax5 = fig.add_axes((0.334, 0.25, 0.2, 0.2), polar=True)
    ax6 = fig.add_axes((0.6, 0.25, 0.2, 0.2), projection='3d')

    ax1.set_title('.polar_plot()')
    ax2.set_title('.flat_plot()')
    ax3.set_title('.plot_3d()')
    ax4.set_title('.plot_color_gradient')
    ax5.set_title('.plot_convex_hull_slice')
    ax6.set_title('TODO: .plot_convex_hull_3d')

    polar_diagram.polar_plot(ws_range=[8, 10, 12, 14, 16], ax=ax1, 
                             colors=('deeppink', 'cyan'),
                             show_legend=False, ls='-', lw=0.8, ms=0.3)
    polar_diagram.flat_plot(ws_range=[8, 10, 12, 14, 16], ax=ax2, 
                            colors=('deeppink', 'cyan'),
                            show_legend=False, ls='-', lw=0.8, ms=0.3)
    polar_diagram.plot_3d(ax=ax3, colors=('deeppink', 'cyan'))
    polar_diagram.plot_color_gradient(ax=ax4, colors=('deeppink', 'cyan'), 
                                      show_legend=False)
    polar_diagram.plot_convex_hull_slice(12, ax=ax5, c='deeppink', 
                                         ls='-', lw=0.8, ms=0.3)
    plt.tight_layout()
    plt.show()
