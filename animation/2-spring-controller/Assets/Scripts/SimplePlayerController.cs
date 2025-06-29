using System;
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

        [SerializeField]
        private float springHalflife = 0.1f;

        private Vector2 velocity = Vector2.zero;
        private Vector2 velocityTarget = Vector2.zero;
        private Vector2 acceleration = Vector2.zero;

        private float deltaTimeAcc = 0f;
        private float playerUpdatePeriod = 0.033f;

        void Start()
        {

        }

        void Update()
        {
            deltaTimeAcc += Time.deltaTime;

            if (deltaTimeAcc >= playerUpdatePeriod)
            {
                /*
                Decouple frequency of frame update and player state update.

                TODO: Handle case when frame rate is too low by performing
                multiple numerical integration steps.
                */

                // DEBUG
                Debug.Log($"Pos: {transform.localPosition}, Vel: {velocity}, Acc: {acceleration}");

                // TODO: Factor out this block into fn


                // Set target velocity according to joystick position
                // TODO: Compare this to adding callback
                Vector2 inputDirection = inputManager.PlayerControls.Player.Move.ReadValue<Vector2>();
                velocityTarget = scaleVelocity * inputDirection;

                (Vector2 deltaPosition, Vector2 newVelocity, Vector2 newAcceleration) = critically_damped_spring_controller_2d(Vector2.zero, velocity, acceleration, velocityTarget, springHalflife, deltaTimeAcc);

                // Update object state
                transform.localPosition += new Vector3(deltaPosition.x, 0, deltaPosition.y);
                velocity = newVelocity;
                acceleration = newAcceleration;


                deltaTimeAcc = 0;

                // TODO: Rotate object based on velocity direction
                if (velocity.magnitude > 0.001 * scaleVelocity)
                {
                    transform.forward = new Vector3(velocity.x, 0, velocity.y).normalized;
                }
            }
        }

        float halflife_to_damping(float halflife, float eps = 1e-5f)
        {
            return (4.0f * 0.69314718056f) / (halflife + eps);
        }

        float fast_negexp(float x)
        {
            return 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
        }

        (Vector2, Vector2, Vector2) critically_damped_spring_controller_2d(
            Vector2 position,
            Vector2 velocity,
            Vector2 acceleration,
            Vector2 velocityTarget,
            float halflife,
            float deltaTime)
        {
            /*
                Applies the force of a critically damped free spring to the displacement of the current velocity to the target velocity.

                Returns the change in position, velocity, and acceleration.

                TODO: Move this function into a physics script
            */

            float lambda = halflife_to_damping(halflife) / 2.0f;
            Vector2 c1 = velocity - velocityTarget;
            Vector2 c2 = acceleration + c1 * lambda;
            float eydt = fast_negexp(lambda * deltaTime);

            Vector2 newPosition = eydt * (((-c2) / (lambda * lambda)) + ((-c1 - c2 * deltaTime) / lambda)) +
                (c2 / (lambda * lambda)) + c1 / lambda + velocityTarget * deltaTime + position;
            Vector2 newVelocity = eydt * (c1 + c2 * deltaTime) + velocityTarget;
            Vector2 newAcceleration = eydt * (acceleration - c2 * lambda * deltaTime);
            return (newPosition, newVelocity, newAcceleration);
        }

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

            // Visualize target velocity
            Handles.color = Color.green;
            Handles.DrawLine(transform.position, transform.position + new Vector3(velocityTarget.x, 0, velocityTarget.y), 2f);

            // Visualize actual velocity
            Handles.color = Color.yellow;
            Handles.DrawLine(transform.position, transform.position + new Vector3(velocity.x, 0, velocity.y), 2f);

            // TODO: Visualize object facing direction
            Handles.color = Color.red;
            Handles.DrawLine(transform.position, transform.position + transform.forward, 2f);
        }
    }
}