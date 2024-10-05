import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))


import torch
from gymnasium.wrappers import RecordVideo
from gyms.cartpole_pendulum import PendulumGym, CartPolePendulumEnv

from dqn_utils import DQN

         
def load_model(model_path, n_observations, n_actions, device):    
    """Load the saved model from the given path."""
    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def visualize_performance(environment, model, device, video_file_name):
    """Record the model's performance in the environment."""
    env = RecordVideo(environment, video_folder="videos", name_prefix=video_file_name)

    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False

    while not done:
        with torch.no_grad():
            action = model(state).max(1)[1].view(1, 1)        
        next_state, reward, terminated, truncated, _  = env.step(action.item())
        done = terminated or truncated        
        state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

    env.close()



    
if __name__ == "__main__":
    
    result_folder_path = 'results'
    if os.path.exists(f"{current_dir}/{result_folder_path}"):
        pass
    else:
        print("No results yet")
        sys.exit()    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PendulumGym(render_mode="rgb_array")


    
    model_files = [
        file for file in os.listdir(f"{current_dir}/{result_folder_path}")
        if file.startswith("policy_net_") and file.endswith(".pth")
    ]    

    for model_file in model_files:
        model_path = os.path.join(f"{current_dir}/{result_folder_path}", model_file)
        
        model = load_model(model_path, env.get_observation_space().shape[0], env.get_action_space().n, device)
        video_file_name = f"performance_{model_file.split('.')[0]}"  # 모델 파일 이름을 영상 이름으로 사용
        visualize_performance(env, model, device, video_file_name)
        print(f"Video for {model_file} saved successfully as {video_file_name}.mp4")
            
