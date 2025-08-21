
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def scenario_probs_fixed(model, mean, std, cost_blue_start=4.1, steps=12):
    prices = np.linspace(cost_blue_start, cost_blue_start-2.5, steps).astype(np.float32)
    car_list, red_list, blue_list = [], [], []
    for cb in prices:
        Xs = np.array([[
            [30, 10, 8],
            [40,  4, 6],
            [41, cb, 6.1],
        ]], dtype=np.float32)
        Xs = (Xs - mean) / std
        Xs_t = torch.tensor(Xs, dtype=torch.float32)
        with torch.no_grad():
            if hasattr(model, "predict_proba"):
                if model.__class__.__name__ == "MNL":
                    P = F.softmax(model(Xs_t), dim=1).detach().numpy()[0]
                else:
                    P = model.predict_proba(Xs_t).detach().numpy()[0]
            else:
                P = F.softmax(model(Xs_t), dim=1).detach().numpy()[0]
        car_list.append(P[0]); red_list.append(P[1]); blue_list.append(P[2])
    return prices, np.array(car_list), np.array(red_list), np.array(blue_list)

prices, car_mnl, red_mnl, blue_mnl   = scenario_probs_fixed(mnl, mean, std)
_,      car_nest, red_nest, blue_nest= scenario_probs_fixed(nest, mean, std)

plt.figure(figsize=(6,4))
plt.plot(prices, car_mnl, label="Car (MNL)")
plt.plot(prices, car_nest, label="Car (Nested)")
plt.xlabel("BlueBus cost (lower = better)")
plt.ylabel("Predicted probability of Car")
plt.title("Car share as BlueBus becomes cheaper")
plt.legend()
plt.tight_layout()
p1="/mnt/data/car_prob_vs_blue_cost.png"
plt.savefig(p1, dpi=150)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(prices, red_mnl, label="RedBus (MNL)")
plt.plot(prices, blue_mnl, label="BlueBus (MNL)")
plt.plot(prices, red_nest, label="RedBus (Nested)")
plt.plot(prices, blue_nest, label="BlueBus (Nested)")
plt.xlabel("BlueBus cost (lower = better)")
plt.ylabel("Predicted probability")
plt.title("Within-nest substitution (Red ↔ Blue)")
plt.legend()
plt.tight_layout()
p2="/mnt/data/bus_probs_vs_blue_cost.png"
plt.savefig(p2, dpi=150)
plt.show()

print(f"[Saved plots]\n- {p1}\n- {p2}")
