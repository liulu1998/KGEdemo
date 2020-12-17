import time
import matplotlib.pyplot as plt


def draw_graph(result: dict, smooth: bool):
    """ 画学习曲线
    :param result: dict, train_model 函数的返回值
    :param smooth: bool, 是否平滑
    """
    def smooth_curve(points, factor=0.8):
        """ 平滑
        """
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    # 横坐标 epochs
    epochs = range(1, result["epochs"] + 1)
    # 平滑
    if smooth:
        for name in ["train_acc", "val_acc", "train_loss", "val_loss"]:
            if name in result:
                result[name] = smooth_curve(result[name])

    if "train_acc" in result:
        plt.plot(epochs, result["train_acc"], 'bo', label='Trianing acc')
        plt.plot(epochs, result["val_acc"], 'b', label='Validation acc')
        plt.title('Training and validation acc')
        plt.legend()
        plt.figure()

    plt.plot(epochs, result["train_loss"], 'bo', label='Train loss')
    plt.plot(epochs, result["val_loss"], 'b', label='Validation loss')
    plt.title('Train and validation loss')
    plt.legend()

    plt.show()
    # plt.imsave(f"./result/plot_totEpoch{result['epochs']}_"
    #            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.png")
