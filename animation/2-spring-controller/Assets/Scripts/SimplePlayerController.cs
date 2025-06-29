using UnityEditor;
using UnityEngine;

namespace StefanWebb
{
    public class SimplePlayerController : MonoBehaviour
    {
        [SerializeField]
        private InputManager inputManager;

        [SerializeField]
        private float scaleVelocity = 1f;

        private Vector2 velocity = Vector2.zero;
        private Vector2 velocityTarget = Vector2.zero;
        private Vector2 acceleration = Vector2.zero;

        void Start()
        {

        }

        void Update()
        {
            // Set velocity according to joystick position
            // TODO: Compare this to adding callback
            Vector2 inputDirection = inputManager.PlayerControls.Player.Move.ReadValue<Vector2>();
            velocity = scaleVelocity * inputDirection;

            // Update object position
            transform.localPosition += Time.deltaTime * new Vector3(velocity.x, 0, velocity.y);

            // TODO: Rotate object based on velocity direction
            if (inputDirection.magnitude > 0.001)
            {
                transform.forward = new Vector3(velocity.x, 0, velocity.y).normalized;
            }
        }

        // void spring_character_update(
        //     float& x,
        //     float& v,
        //     float& a,
        //     float v_goal,
        //     float halflife,
        //     float dt)
        // {
        //     float y = halflife_to_damping(halflife) / 2.0f;
        //     float j0 = v - v_goal;
        //     float j1 = a + j0 * y;
        //     float eydt = fast_negexp(y * dt);

        //     x = eydt * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y)) +
        //         (j1 / (y * y)) + j0 / y + v_goal * dt + x;
        //     v = eydt * (j0 + j1 * dt) + v_goal;
        //     a = eydt * (a - j1 * y * dt);
        // }

        // void FixedUpdate()
        // {
        //     // always draw a 5-unit colored line from the origin
        //     Color color = new Color(0, 0, 1.0f);
        //     Debug.DrawLine(transform.position, transform.position + 1 * transform.forward, color);
        // }

        // Visualize the character position and velocity
        void OnDrawGizmos()
        {
            // Gizmos.color = Color.yellow;
            // Gizmos.DrawSphere(transform.position, 1);
            // Handles.color = Color.yellow;
            Handles.DrawWireDisc(transform.position, new Vector3(0, 1, 0), 1, 2f);

            // Handles.ArrowHandleCap(0, transform.position, transform.rotation, 0.5f, EventType.Repaint);

            // Visualize velocity
            Handles.color = Color.yellow;
            Handles.DrawLine(transform.position, transform.position + new Vector3(velocity.x, 0, velocity.y), 2f);

            // TODO: Visualize object facing direction
            Handles.color = Color.red;
            Handles.DrawLine(transform.position, transform.position + transform.forward, 2f);
        }
    }
}