import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. load data
# use forward slashes so Windows doesn't complain
df = pd.read_csv("smash_analysis/smash_coordinates.csv")

# 2. feature from statement: "based on locations of birdie and racket"
# birdie -> landing_x, landing_y
# racket/player -> player_location_x, player_location_y
df["distance_to_birdie"] = np.sqrt(
    (df["landing_x"] - df["player_location_x"])**2 +
    (df["landing_y"] - df["player_location_y"])**2
)

# 3. proxy label for "smash success"
# NOTE: replace this rule with real labels when you have them
df["smash_success"] = np.where(
    (df["landing_y"] > 500) & (df["distance_to_birdie"] < 250),
    1,
    0
)

# 4. visualize: where smashes are more successful
plt.figure(figsize=(6,5))
sns.scatterplot(
    data=df,
    x="landing_x",
    y="landing_y",
    hue="smash_success",
    alpha=0.7
)
plt.title("Predicted Smash Success by Shuttle Location")
plt.xlabel("Landing X")
plt.ylabel("Landing Y")
plt.tight_layout()
plt.show()

# 5. visualize: does racket position matter?
plt.figure(figsize=(6,5))
sns.scatterplot(
    data=df,
    x="player_location_x",
    y="player_location_y",
    hue="smash_success",
    alpha=0.7
)
plt.title("Predicted Smash Success by Player (Racket) Location")
plt.xlabel("Player X")
plt.ylabel("Player Y")
plt.tight_layout()
plt.show()
