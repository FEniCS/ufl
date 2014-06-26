
class A:
    def _repr_png_(self):
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from([1, 2,3])
        G.add_edges_from([(1,2),(1,3), (2,3)])

        fig = plt.figure()
        nx.draw_spring(G)

        #fig, ax = plt.subplots()

        from IPython.core.pylabtools import print_figure
        data = print_figure(fig, 'png')
        # We MUST close the figure, otherwise IPython's display machinery
        # will pick it up and send it as output, resulting in a double display
        plt.close(fig)
        return data

if 0:
    # Other way found on internet:
    import networkx as nx
    import matplotlib.pyplot as plt
    import io
    from matplotlib.figure import Figure

    class MyGraph(nx.Graph):
        def _repr_svg_(self):
            plt.ioff() # turn off interactive mode
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            nx.draw_shell(self, ax=ax)
            output = io.StringIO()
            fig.savefig(output,format='svg')
            plt.ion() # turn on interactive mode
            return output.getvalue()

