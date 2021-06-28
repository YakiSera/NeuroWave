from matplotlib import pyplot as plt

def start_graph():
    plt.figure(figsize=(7, 6))
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.grid(True)

def add_graph(x, y, name, color):
    plt.plot(x, y, color, label=name)

def finish_graph():
    plt.legend(fontsize=14)
    plt.show()