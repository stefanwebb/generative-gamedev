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

        // TODO: How does this relate to "angular velocity"?
        private Quaternion forwardTarget = Quaternion.identity; // new Vector2(0, 1f);
        private Quaternion forwardVelocity = new Quaternion(0, 0, 0, 0); // Vector2.zero;

        // NOTE: pastPosition stores local position
        // whereas futurePosition is offset relative to current
        private int startIdxPastPosition = 0;
        private Vector2[] pastPosition = new Vector2[30];
        private Vector2[] futurePosition = new Vector2[30];


        private float deltaTimeAcc = 0f;
        private float playerUpdatePeriod = 0.033f;

        void Start()
        {
            // TODO: Way to do this during intialization?
            for (int i = 0; i < 30; i++)
            {
                pastPosition[i] = transform.localPosition;
                futurePosition[i] = Vector2.zero;
            }
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

                // DEBUG: Store previous second of movement
                // TODO: Can I move this to OnDrawGizmos?
                pastPosition[startIdxPastPosition].x = transform.position.x;
                pastPosition[startIdxPastPosition].y = transform.position.z;
                startIdxPastPosition++;

                if (startIdxPastPosition >= 30)
                {
                    startIdxPastPosition = 0;
                }

                // DEBUG
                // Debug.Log($"Pos: {transform.localPosition}, Vel: {velocity}, Acc: {acceleration}");

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

                // Update target forward facing direction
                if (velocityTarget.magnitude > 0.001 * scaleVelocity)
                {
                    forwardTarget = Quaternion.LookRotation(new Vector3(velocity.x, 0, velocity.y).normalized, Vector3.up);
                    // transform.forward = forwardTarget * Vector3.forward;
                }

                transform.forward = Vector3.Slerp(transform.forward, forwardTarget * Vector3.forward, Math.Min(deltaTimeAcc / 0.075f, 1f));

                deltaTimeAcc = 0;

                // Update actual forward facing direction
                // (Vector2 newForward, Vector2 newForwardVelocity) = critically_damped_spring_2d(transform.forward, velocity, forwardTarget, springHalflife, deltaTimeAcc);

                // transform.forward = new Vector3(newForward.x, 0, newForward.y).normalized;

                // forwardVelocity = newForwardVelocity;
            }

            // DEBUG: Predict next second of movement
            // TODO: Does this actually work?
            // futurePosition[0] = new Vector3(deltaPosition.x, 0, deltaPosition.y);
            for (int i = 0; i < 30; i++)
            {
                (futurePosition[i], _, _) = critically_damped_spring_controller_2d(Vector2.zero, velocity, acceleration, velocityTarget, springHalflife, 1f / 30f * i);
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

            // TODO: Check that this is a stable int method
            // Should position and velocity calcs be switched?
            Vector2 newPosition = eydt * (((-c2) / (lambda * lambda)) + ((-c1 - c2 * deltaTime) / lambda)) +
                (c2 / (lambda * lambda)) + c1 / lambda + velocityTarget * deltaTime + position;
            Vector2 newVelocity = eydt * (c1 + c2 * deltaTime) + velocityTarget;
            Vector2 newAcceleration = eydt * (acceleration - c2 * lambda * deltaTime);
            return (newPosition, newVelocity, newAcceleration);
        }

        (Vector2, Vector2) critically_damped_spring_2d(
            Vector2 position,
            Vector2 velocity,
            Vector2 positionTarget,
            float halflife,
            float deltaTime)
        {
            /*
                Applies the force of a critically damped free spring to the
                displayment of an arbitrary vector to a target vector.

                Returns the change in position and velocity.

                TODO: Move this function into a physics script
            */

            float lambda = halflife_to_damping(halflife) / 2.0f;
            Vector2 c1 = position - positionTarget;
            Vector2 c2 = velocity + c1 * lambda;
            float eydt = fast_negexp(lambda * deltaTime);

            // TODO: Check that this is a stable int method
            // Should position and velocity calcs be switched?

            Vector2 newPosition = eydt * (c1 + c2 * deltaTime) + positionTarget;
            Vector2 newVelocity = eydt * (velocity - c2 * lambda * deltaTime);
            return (newPosition, newVelocity);
        }

        // (Vector2, Vector2) critically_damped_spring_quaternion(
        //     Quaternion rotation,
        //     Quaternion velocity,
        //     Quaternion rotationTarget,
        //     float halflife,
        //     float deltaTime)
        // {
        //     /*
        //         Applies the force of a critically damped free spring to the
        //         displayment of an arbitrary vector to a target vector.

        //         Returns the change in position and velocity.

        //         TODO: Move this function into a physics script
        //     */

        //     float lambda = halflife_to_damping(halflife) / 2.0f;
        //     Vector2 c1 = rotation - rotationTarget;
        //     Vector2 c2 = velocity + c1 * lambda;
        //     float eydt = fast_negexp(lambda * deltaTime);

        //     // TODO: Check that this is a stable int method
        //     // Should position and velocity calcs be switched?

        //     Vector2 newPosition = eydt * (c1 + c2 * deltaTime) + rotationTarget;
        //     Vector2 newVelocity = eydt * (velocity - c2 * lambda * deltaTime);
        //     return (newPosition, newVelocity);
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

            // Visualize predicted path
            // TODO: Use draw lines method?
            Handles.color = Color.white;
            for (int i = 1; i < 30; i++)
            {
                Handles.DrawLine(transform.position + new Vector3(futurePosition[i - 1].x, 0, futurePosition[i - 1].y), transform.position + new Vector3(futurePosition[i].x, 0, futurePosition[i].y), 2f);
            }

            // Visualize past path
            // TODO: Use draw lines method?
            // TODO: Don't hardcode 30 segments
            Handles.color = Color.purple;

            for (int i = 0; i < 29; i++)
            {
                int fromIdx2 = (30 + startIdxPastPosition + i) % 30;
                int toIdx = (30 + startIdxPastPosition + i + 1) % 30;

                Handles.DrawLine(new Vector3(pastPosition[fromIdx2].x, 0, pastPosition[fromIdx2].y), new Vector3(pastPosition[toIdx].x, 0, pastPosition[toIdx].y), 2f);
            }

            // DEBUG: Checking where past positions are

            // Debug.Log($"{(30 + startIdxPastPosition - 1) % 30}");
            int fromIdx = (30 + startIdxPastPosition - 1) % 30;
            Handles.DrawWireDisc(new Vector3(pastPosition[fromIdx].x, 0, pastPosition[fromIdx].y), new Vector3(0, 1, 0), 0.25f, 2f);

            Handles.DrawWireDisc(new Vector3(transform.position.x, 0, transform.position.z), new Vector3(0, 1, 0), 0.25f, 2f);

            Handles.DrawLine(new Vector3(transform.position.x, 0, transform.position.z), new Vector3(pastPosition[fromIdx].x, 0, pastPosition[fromIdx].y), 2f);


            // TODO: Must be able to always get current position into past position array
            // int firstIdxPastPosition = (30 + startIdxPastPosition - 1) % 30;
            // Handles.DrawLine(new Vector3(pastPosition[firstIdxPastPosition].x, 0, pastPosition[firstIdxPastPosition].y), new Vector3(transform.position.x, 0, transform.position.y), 2f);

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