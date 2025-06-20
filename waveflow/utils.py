import matplotlib.pyplot as plt
def show_comparison(pred, true, title=""):
    plt.subplot(1,2,1); plt.title("Ground-truth"); plt.imshow(true,origin="lower",aspect="auto")
    plt.subplot(1,2,2); plt.title("Reconstruido"+title); plt.imshow(pred.T,origin="lower",aspect="auto")
    plt.tight_layout(); plt.show()
