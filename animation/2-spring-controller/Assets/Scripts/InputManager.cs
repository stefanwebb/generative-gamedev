using System.Runtime.CompilerServices;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

namespace StefanWebb
{
    // TODO: Separate out callbacks from this class
    // TODO: When adding polish, add additional controller displays
    public class InputManager : MonoBehaviour, PlayerControls.IPlayerActions
    {
        [Header("Colors")]
        [SerializeField] private Color colorOverlay;
        [SerializeField] private Color colorActive;
        [SerializeField] private Color colorInactive;
        [SerializeField] private Color colorButtonA;
        [SerializeField] private Color colorButtonB;
        [SerializeField] private Color colorButtonX;
        [SerializeField] private Color colorButtonY;
        [SerializeField] private Color colorButtonXbox;
        [SerializeField] private Color colorNotFound;

        // TODO: Separate out transparency from color

        // TODO: Layout params like size, margin, corner of screen, stick radius etc.
        [Header("Layout")]
        [SerializeField] private float stickRadius;
        [SerializeField] private float stickEpsilon;

        [Header("UI Elements")]
        [SerializeField] private Image overlay;
        [SerializeField] private Image buttonA;
        [SerializeField] private Image buttonB;
        [SerializeField] private Image buttonX;
        [SerializeField] private Image buttonY;
        [SerializeField] private Image buttonView;
        [SerializeField] private Image buttonMenu;
        [SerializeField] private Image buttonShare;
        [SerializeField] private Image buttonXbox;
        [SerializeField] private Image stickLeft;
        [SerializeField] private Image stickRight;
        [SerializeField] private TextMeshProUGUI msgNotFound;

        // TODO: Bumpers and triggers

        // TODO: Directional-Pad
        // [SerializeField] private Image dPad;

        public PlayerControls PlayerControls;

        private Vector2 moveDirection = Vector2.zero;

        // private GameObject stickLeftGameObject;
        // private GameObject stickRightGameObject;

        // private void Awake()
        // {

        // }

        private void Start()
        {
            // DEBUG: List gamepads
            bool xBoxControllerFound = false;
            if (Gamepad.all.Count > 0)
            {
                foreach (Gamepad gamepad in Gamepad.all)
                {
                    if (gamepad.name == "XboxGamepadMacOSWireless")
                    {
                        xBoxControllerFound = true;
                    }
                    Debug.Log($"Gamepad found: {gamepad.name}");
                }
            }

            else
            {
                Debug.Log($"No gamepads found");
            }

            // TODO: Switch off overlay if no gamepad found
            // Or maybe change the overlay to an error color
            if (xBoxControllerFound)
            {
                OnFound();
            }
            else
            {
                OnNotFound();
            }
        }

        // private void Start()
        // {

        // }

        private void OnFound()
        {
            // TODO: Tidy with multiple assignment
            msgNotFound.enabled = false;
            overlay.color = colorOverlay;
            buttonB.color = colorInactive;
            buttonX.color = colorInactive;
            buttonY.color = colorInactive;
            buttonView.color = colorInactive;
            buttonMenu.color = colorInactive;
            buttonShare.color = colorInactive;
            buttonXbox.color = colorInactive;
            stickLeft.color = colorInactive;
            stickRight.color = colorInactive;
        }

        private void OnNotFound()
        {
            // TODO: Tidy with multiple assignment
            msgNotFound.enabled = true;
            overlay.color = buttonA.color = colorNotFound;
            buttonB.color = colorNotFound;
            buttonX.color = colorNotFound;
            buttonY.color = colorNotFound;
            buttonView.color = colorNotFound;
            buttonMenu.color = colorNotFound;
            buttonShare.color = colorNotFound;
            buttonXbox.color = colorNotFound;
            stickLeft.color = colorNotFound;
            stickRight.color = colorNotFound;
            PlayerControls.Player.RemoveCallbacks(this);
        }

        private void OnEnable()
        {
            PlayerControls = new PlayerControls();
            PlayerControls.Enable();

            PlayerControls.Player.Enable();
            PlayerControls.Player.SetCallbacks(this);
        }

        private void OnDisable()
        {
            PlayerControls.Player.Disable();
            PlayerControls.Player.RemoveCallbacks(this);
        }

        public void OnMove(InputAction.CallbackContext context)
        {
            // Is this inefficient to do in OnMove?
            RectTransform rectTransform = stickLeft.gameObject.transform as RectTransform;

            Vector2 movementInput = context.ReadValue<Vector2>();
            Vector3 newPosition = new Vector3(stickRadius * movementInput.x, stickRadius * movementInput.y, 0);
            rectTransform.anchoredPosition3D = newPosition;

            // TODO: Slightly more sophisticated active detection based on smoothed joystick velocity 
            if (movementInput.magnitude < stickEpsilon) // && stickLeft.color != colorInactive)
            {
                stickLeft.color = colorInactive;
            }
            else
            {
                stickLeft.color = colorActive;
            }
        }

        public void OnLook(InputAction.CallbackContext context)
        {
            // Is this inefficient to do in OnMove?
            RectTransform rectTransform = stickRight.gameObject.transform as RectTransform;

            Vector2 movementInput = context.ReadValue<Vector2>();
            Vector3 newPosition = new Vector3(stickRadius * movementInput.x, stickRadius * movementInput.y, 0);
            rectTransform.anchoredPosition3D = newPosition;

            // TODO: Slightly more sophisticated active detection based on smoothed joystick velocity
            if (movementInput.magnitude < stickEpsilon) // && stickRight.color != colorInactive)
            {
                stickRight.color = colorInactive;
            }
            else
            {
                stickRight.color = colorActive;
            }
        }

        public void OnNavigate(InputAction.CallbackContext context)
        {
            // Directional-pad
            Debug.Log("Navigate");
        }

        public void OnAction(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                buttonA.color = colorButtonA;
            }
            else if (context.canceled)
            {
                buttonA.color = colorInactive;
            }
        }

        public void OnSecondaryAction(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                buttonX.color = colorButtonX;
            }
            else if (context.canceled)
            {
                buttonX.color = colorInactive;
            }
        }

        public void OnInteract(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                buttonB.color = colorButtonB;
            }
            else if (context.canceled)
            {
                buttonB.color = colorInactive;
            }
        }

        public void OnSecondaryInteract(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                buttonY.color = colorButtonY;
            }
            else if (context.canceled)
            {
                buttonY.color = colorInactive;
            }
        }

        public void OnSystem(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                buttonXbox.color = colorButtonXbox;
            }
            else if (context.canceled)
            {
                buttonXbox.color = colorInactive;
            }
        }

        public void OnView(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                buttonView.color = colorActive;
            }
            else if (context.canceled)
            {
                buttonView.color = colorInactive;
            }
        }

        public void OnShare(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                buttonShare.color = colorActive;
            }
            else if (context.canceled)
            {
                buttonShare.color = colorInactive;
            }
        }

        public void OnMenu(InputAction.CallbackContext context)
        {
            if (context.performed)
            {
                buttonMenu.color = colorActive;
            }
            else if (context.canceled)
            {
                buttonMenu.color = colorInactive;
            }
        }

        public void OnModifyLeft(InputAction.CallbackContext context)
        {
            // if (context.performed)
            // {
            //     buttonBumperLeft.color = colorActive;
            // }
            // else if (context.canceled)
            // {
            //     buttonBumperLeft.color = colorInactive;
            // }
        }

        public void OnModifyRight(InputAction.CallbackContext context)
        {
            Debug.Log("ModifyRight");
        }

        public void OnTriggerLeft(InputAction.CallbackContext context)
        {
            Debug.Log("TriggerLeft");
        }

        public void OnTriggerRight(InputAction.CallbackContext context)
        {
            Debug.Log("TriggerRight");
        }
    }
}
