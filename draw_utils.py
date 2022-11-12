import matplotlib.pyplot as plt
import numpy as np

# stats - [(имя значения, значение)]
def draw_stats(ax, pipe, stats):
    for i, s in enumerate(stats):
        m = s + ': ' + f'{pipe.val(s):.2f}'
        ax.text(0.05, 0.9 - 0.1 * i, m, color='r', backgroundcolor='1', alpha=0.8, transform=ax.transAxes)

# to_draw - [пайплайн]
# draw_f - функция отрисовки
def draw_plots(to_draw, draw_f, title='', max_cols=2):
    if(type(draw_f) != list):
        draw_f = [draw_f]
    cols = min(len(to_draw), 2)
    cols = max(cols, len(draw_f))
    cols = min(cols, max_cols)

    rows = int(np.ceil(len(to_draw) * len(draw_f) / cols))
    height = rows * 3

    plt.subplots(rows, cols, figsize=(16, height))
    if(len(title)):
        plt.suptitle(title)
    
    i = 0
    for _, pipe in enumerate(to_draw):
        for f in draw_f:
            i = i + 1
            ax = plt.subplot(rows, cols, i)
            plt.title(pipe.name)
            f(ax, pipe)
            plt.legend(loc='lower right')

def draw_metric(metrics=[]):
    def do_draw_metric(ax, p):
        for metric in metrics:
            history = p.history(metric)
            ax.plot(range(len(history)), history, label=metric)
    return do_draw_metric

def draw_compare_stats(nstats=[], labels=[]):
    def do_draw_compare_stats(ax, p):
        stats = []
        if(type(nstats) == list and type(nstats[0]) == 'str'):
            for s in nstats:
                stats.append(p.val(s))
        else:
            stats = p.val(nstats)
            if(type(stats) == np.matrix):
                stats = np.asarray(stats).reshape(-1)

        numbers = np.arange(0,len(stats))
        if(len(labels) <= len(stats)):
            gen_labels_count = len(stats) - len(labels)
            tick_labels = labels + [str(num) for num in np.arange(0, gen_labels_count)]
        else:
            tick_labels = labels

        cc=['']*len(numbers)
        for n,val in enumerate(stats):
            if val<0:
                cc[n]='red'
            elif val>=0:
                cc[n]='blue'

        ax.bar(x = numbers, height = stats, label=str(nstats), color=cc)
        plt.xticks(np.arange(0,len(stats)), tick_labels)

    return do_draw_compare_stats