# backend/intent_parser.py
"""
Intent Parser - Analyzes user input and extracts intent
No LLM needed - uses keyword matching and patterns
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Intent:
    """Represents the parsed user intent"""
    primary_action: str  # create, add, modify, remove
    ui_type: str  # dashboard, form, table, card, etc.
    components: List[str]  # List of component names
    layout: str  # grid, flex, stack, etc.
    modifiers: Dict[str, any]  # Additional properties
    confidence: float  # 0-1 confidence score

class IntentParser:
    """Parses user input to determine what UI to build"""
    
    # Keyword mappings
    ACTIONS = {
        'create': ['create', 'make', 'build', 'generate', 'new'],
        'add': ['add', 'include', 'insert', 'put'],
        'modify': ['change', 'update', 'modify', 'edit', 'alter'],
        'remove': ['remove', 'delete', 'take out', 'get rid of'],
    }
    
    UI_TYPES = {
        'dashboard': ['dashboard', 'overview', 'summary', 'analytics'],
        'form': ['form', 'input form', 'registration', 'signup', 'login'],
        'table': ['table', 'data table', 'grid', 'list'],
        'card': ['card', 'info card', 'profile card'],
        'navbar': ['navbar', 'navigation', 'menu', 'header'],
        'sidebar': ['sidebar', 'side menu', 'drawer'],
        'modal': ['modal', 'dialog', 'popup', 'overlay'],
        'button': ['button', 'btn', 'action button'],
        'chart': ['chart', 'graph', 'visualization', 'plot'],
    }
    
    COMPONENTS = {
        'Button': ['button', 'btn', 'submit', 'action'],
        'Card': ['card', 'panel', 'box'],
        'Input': ['input', 'textbox', 'field', 'text field'],
        'Table': ['table', 'data table', 'grid'],
        'Chart': ['chart', 'graph', 'bar chart', 'line chart', 'pie chart'],
        'Navbar': ['navbar', 'navigation', 'top bar', 'header'],
        'Sidebar': ['sidebar', 'side panel', 'drawer'],
        'Modal': ['modal', 'dialog', 'popup'],
    }
    
    LAYOUTS = {
        'grid': ['grid', 'tiles', 'cards'],
        'flex': ['flex', 'flexible', 'row', 'column'],
        'stack': ['stack', 'vertical', 'horizontal'],
        'split': ['split', 'divided', 'sections'],
    }
    
    VARIANTS = {
        'primary': ['primary', 'main', 'important'],
        'secondary': ['secondary', 'alternate'],
        'outline': ['outline', 'outlined', 'border'],
        'ghost': ['ghost', 'transparent', 'minimal'],
    }
    
    COLORS = {
        'blue': ['blue'],
        'red': ['red', 'danger', 'error'],
        'green': ['green', 'success'],
        'yellow': ['yellow', 'warning'],
        'gray': ['gray', 'grey', 'neutral'],
    }
    
    def __init__(self):
        """Initialize the intent parser"""
        pass
    
    def parse(self, user_input: str) -> Intent:
        """
        Parse user input and return intent
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Intent object with parsed information
        """
        user_input = user_input.lower().strip()
        
        # Extract action
        action = self._extract_action(user_input)
        
        # Extract UI type
        ui_type = self._extract_ui_type(user_input)
        
        # Extract components
        components = self._extract_components(user_input)
        
        # Extract layout
        layout = self._extract_layout(user_input)
        
        # Extract modifiers (variant, color, etc.)
        modifiers = self._extract_modifiers(user_input)
        
        # Calculate confidence
        confidence = self._calculate_confidence(action, ui_type, components)
        
        return Intent(
            primary_action=action,
            ui_type=ui_type,
            components=components,
            layout=layout,
            modifiers=modifiers,
            confidence=confidence
        )
    
    def _extract_action(self, text: str) -> str:
        """Extract the primary action from text"""
        for action, keywords in self.ACTIONS.items():
            if any(keyword in text for keyword in keywords):
                return action
        return 'create'  # Default action
    
    def _extract_ui_type(self, text: str) -> str:
        """Extract the UI type from text"""
        for ui_type, keywords in self.UI_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return ui_type
        
        # If no specific type found, infer from components
        if 'input' in text and 'button' in text:
            return 'form'
        elif 'card' in text or 'kpi' in text:
            return 'dashboard'
        
        return 'generic'  # Default
    
    def _extract_components(self, text: str) -> List[str]:
        """Extract component names from text"""
        found_components = []
        
        for component, keywords in self.COMPONENTS.items():
            if any(keyword in text for keyword in keywords):
                if component not in found_components:
                    found_components.append(component)
        
        # If no components found, infer from UI type
        if not found_components:
            found_components = self._infer_components_from_ui_type(
                self._extract_ui_type(text)
            )
        
        return found_components
    
    def _extract_layout(self, text: str) -> str:
        """Extract layout type from text"""
        for layout, keywords in self.LAYOUTS.items():
            if any(keyword in text for keyword in keywords):
                return layout
        return 'flex'  # Default layout
    
    def _extract_modifiers(self, text: str) -> Dict[str, any]:
        """Extract additional modifiers (variant, color, etc.)"""
        modifiers = {}
        
        # Extract variant
        for variant, keywords in self.VARIANTS.items():
            if any(keyword in text for keyword in keywords):
                modifiers['variant'] = variant
                break
        
        # Extract color
        for color, keywords in self.COLORS.items():
            if any(keyword in text for keyword in keywords):
                modifiers['color'] = color
                break
        
        # Extract size
        if 'large' in text or 'big' in text:
            modifiers['size'] = 'large'
        elif 'small' in text or 'tiny' in text:
            modifiers['size'] = 'small'
        
        # Extract numbers (for tables, charts, etc.)
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            modifiers['count'] = int(numbers[0])
        
        return modifiers
    
    def _infer_components_from_ui_type(self, ui_type: str) -> List[str]:
        """Infer default components based on UI type"""
        defaults = {
            'dashboard': ['Card', 'Chart'],
            'form': ['Input', 'Button'],
            'table': ['Table'],
            'card': ['Card'],
            'navbar': ['Navbar'],
            'sidebar': ['Sidebar'],
            'modal': ['Modal', 'Button'],
            'button': ['Button'],
            'chart': ['Chart'],
        }
        return defaults.get(ui_type, ['Card'])
    
    def _calculate_confidence(self, action: str, ui_type: str, 
                            components: List[str]) -> float:
        """Calculate confidence score for the parsed intent"""
        score = 0.5  # Base score
        
        if action != 'create':
            score += 0.1
        
        if ui_type != 'generic':
            score += 0.2
        
        if components:
            score += 0.2
        
        return min(score, 1.0)


# Example usage
if __name__ == '__main__':
    parser = IntentParser()
    
    test_inputs = [
        "Create a dashboard with cards and charts",
        "Build a login form with email and password",
        "Make a data table with 5 columns",
        "Add a button",
        "Create a navbar with links",
    ]
    
    for input_text in test_inputs:
        intent = parser.parse(input_text)
        print(f"\nInput: {input_text}")
        print(f"Intent: {intent}")