/*
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International License
https://creativecommons.org/licenses/by-sa/4.0/deed.en

*/
using UnityEngine;
using Unity.Mathematics;

namespace StefanWebb
{
    public static class SpringPhysics
    {
        public static float halflife_to_damping(float halflife, float eps = 1e-5f)
        {
            return (4.0f * 0.69314718056f) / (halflife + eps);
        }

        public static float fast_negexp(float x)
        {
            return 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
        }

        public static (Vector2, Vector2, Vector2) critically_damped_velocity(
            Vector2 position,
            Vector2 velocity,
            Vector2 acceleration,
            Vector2 velocityTarget,
            float lambda,
            float deltaTime)
        {
            /*
                Applies the force of a critically damped free spring to the displacement of the current velocity to the target velocity.

                Returns the change in position, velocity, and acceleration.
            */

            // float lambda = halflife_to_damping(halflife) / 2.0f;
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

        public static (Vector3, Vector3, Vector3) critically_damped_velocity(
            Vector3 position,
            Vector3 velocity,
            Vector3 acceleration,
            Vector3 velocityTarget,
            float lambda,
            float deltaTime)
        {
            /*
                Applies physics of a critically damped free spring to the displacement of the current velocity to the target velocity.

                Returns the change in position, velocity, and acceleration.
            */

            // float lambda = halflife_to_damping(halflife) / 2.0f;
            Vector3 c1 = velocity - velocityTarget;
            Vector3 c2 = acceleration + c1 * lambda;
            float eydt = fast_negexp(lambda * deltaTime);

            // TODO: Check that this is a stable int method
            // Should position and velocity calcs be switched?
            Vector3 newPosition = eydt * (((-c2) / (lambda * lambda)) + ((-c1 - c2 * deltaTime) / lambda)) +
                (c2 / (lambda * lambda)) + c1 / lambda + velocityTarget * deltaTime + position;
            Vector3 newVelocity = eydt * (c1 + c2 * deltaTime) + velocityTarget;
            Vector3 newAcceleration = eydt * (acceleration - c2 * lambda * deltaTime);
            return (newPosition, newVelocity, newAcceleration);
        }

        public static (quaternion, float3) critically_damped_rotation(
            quaternion rotation,
            float3 angularVelocity,
            quaternion rotationTarget,
            float lambda,
            float deltaTime)
        {
            /*
                Applies physics of a critically damped spring to the difference between
                two rotations.
            */

            float3 c1 = MathExtensions.QuaternionToScaledAngleAxis(MathExtensions.Abs(math.mul(rotation, math.inverse(rotationTarget))));
            float3 c2 = angularVelocity + c1 * lambda;
            float eydt = fast_negexp(lambda * deltaTime); // this could be precomputed if several agents use it the same frame

            rotation = math.mul(MathExtensions.QuaternionFromScaledAngleAxis(eydt * (c1 + c2 * deltaTime)), rotationTarget);
            angularVelocity = eydt * (angularVelocity - c2 * lambda * deltaTime);

            return (rotation, angularVelocity);
        }


    }
}