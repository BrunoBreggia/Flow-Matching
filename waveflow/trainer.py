import torch
import torch.nn.functional as F
from tqdm import tqdm

from pathlib import Path
from .model import MLPVectorField
from .paths import WaveConditionalPath
from .data import WaveTensorDataset


class ConditionalFMT:
    """
    Entrenador Flow-Matching condicional totalmente parametrizable.
    """

    def __init__(
        self,
        path: WaveConditionalPath,
        model: MLPVectorField,
        batch_size: int = 1024,
        eps_t: float = 1e-3,
    ):
        self.path, self.model = path, model
        self.batch_size, self.eps_t = batch_size, eps_t

    # ---------- un paso de gradiente ----------
    def _train_step(self):
        u_star, coords = self.path.sample_conditioning_variable(self.batch_size)
        t = torch.rand(self.batch_size, 1, device=u_star.device)
        t = t * (1 - 2 * self.eps_t) + self.eps_t  # evita 0 y 1

        x_t = self.path.sample_conditional_path((u_star, coords), t)
        u_ref = self.path.conditional_vector_field(x_t, (u_star, coords), t)
        u_pred = self.model(x_t, coords, t)

        return F.mse_loss(u_pred, u_ref)

    # ---------- bucle de entrenamiento ----------
    def train(
        self,
        steps: int = 30_000,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
        grad_clip: float | None = 1.0,
    ):
        device = torch.device(device)
        self.model.to(device).train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        pbar = tqdm(range(steps))
        for step in pbar:
            opt.zero_grad()
            loss = self._train_step()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            opt.step()
            pbar.set_description(f"step {step:06d}  loss={loss.item():.3e}")

        self.model.eval()


# ===============================================================
# FUNCIÓN DE ALTO NIVEL
# ===============================================================


def train_flow(
    pt_file: Path | str,
    *,
    batch_size: int = 1024,
    steps: int = 30_000,
    lr: float = 1e-3,
    device: torch.device | str = "cuda",
    eps_t: float = 1e-3,
) -> Path:
    """
    Entrena un vector-field para la solución de ondas y guarda el modelo.

    Returns
    -------
    Path  : ruta al archivo *.fm.pt
    """
    pt_file = Path(pt_file)
    ds = WaveTensorDataset(pt_file)
    path = WaveConditionalPath(ds)
    net = MLPVectorField()

    trainer = ConditionalFMT(
        path,
        net,
        batch_size=batch_size,
        eps_t=eps_t,
    )
    trainer.train(steps=steps, lr=lr, device=device)

    model_dir = pt_file.parent / "flow_models"
    model_dir.mkdir(exist_ok=True)
    model_file = model_dir / pt_file.with_suffix(".fm.pt").name
    torch.save(net.state_dict(), model_file)
    print(f"✅ modelo guardado → {model_file}")
    return model_file
