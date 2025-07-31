import sys
import copy
import config

# Add the parent directory to the path to allow importing GameModel
sys.path.append(".." )
from GameModel.TempleOfHorror import TempleOfHorror


def test_environment_isolation():
    """
    Tests that the environment state is properly isolated across recursive calls.
    """
    print("Running test: test_environment_isolation" )

    # 1. Create and initialize the master environment
    original_env = TempleOfHorror()
    original_env.reset()

    # 2. Simulate a game action to change the initial state
    # Player 0 provides the first message (e.g., "No message")
    original_env.step_message(config.message_space[0])
    print(f"Parent env state before recursive call: turn={original_env.provide_message}, history={list(original_env.message_history)}" )

    # 3. Save the critical state of the original environment before the recursive call
    saved_state = {
        "provide_message": original_env.provide_message,
        "message_history": list(original_env.message_history)
    }

    # 4. Define a mock recursive function that simulates a deeper game state
    def mock_recursive_call(env_instance):
        print("  -> Entering mock recursive call..." )
        # This function now works on a separate 'env_instance'
        # Player 1 provides the second message (e.g., "I have fire" )
        env_instance.step_message(config.message_space[1])
        print(f"     Inner env state has been modified: turn={env_instance.provide_message}, history={list(env_instance.message_history)}" )
        print("  <- Exiting mock recursive call..." )

    # 5. Call the mock recursive function with a DEEP COPY of the original environment
    #    This simulates passing the environment down one level in the recursion
    recursive_env_copy = copy.deepcopy(original_env)
    mock_recursive_call(recursive_env_copy)

    # 6. Get the state of the original environment AFTER the recursive call has finished
    state_after_call = {
        "provide_message": original_env.provide_message,
        "message_history": list(original_env.message_history)
    }
    print(f"Parent env state after recursive call: turn={state_after_call['provide_message']}, history={state_after_call['message_history']}" )

    # 7. Assert that the original environment's state has NOT changed
    assert saved_state["provide_message"] == state_after_call["provide_message"], "Test Failed: 'provide_message' was modified!"
    assert saved_state["message_history"] == state_after_call["message_history"], "Test Failed: 'message_history' was modified!"

    print("\n[SUCCESS] Test passed. The original environment's state was preserved correctly." )


if __name__ == "__main__":
    test_environment_isolation()
