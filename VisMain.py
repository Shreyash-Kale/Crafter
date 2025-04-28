# main.py - Main application entry point for VisGUI system

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSplitter, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QStackedWidget, QPushButton
from PyQt5.QtWidgets import QCheckBox

from VideoPlayer import VideoPlayerWidget
from Vis import VisualizationWidget, InfoPanel
from VisDataManager import DataManager
from VisTimelineController import TimelineController
from VisConfig import DEFAULT_LOG_DIR, RESULTS_LOG_DIR, VIZ_COLORS, DEFAULT_FPS
import random

class MainWindow(QMainWindow):
    """Main application window containing video player and visualization panels"""
    
    def __init__(self):
        super().__init__()
        
        # Setup the main window properties
        self.setWindowTitle("Crafter Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.frame_offset = 1

        # Create the central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.showing_info_panel = False  # Track current view

        
        # Create a splitter to divide the window into two panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.toggle_button = QPushButton("Toggle Info/Plots")
        main_layout.addWidget(self.toggle_button)

        self.toggle_button.clicked.connect(self.toggle_view)

        self.showing_info_panel = False  # Track what is shown

        
        # Create data manager to handle CSV data and synchronization
        self.data_manager = DataManager()
        
        # Create video player widget (left panel)
        self.video_player = VideoPlayerWidget()
        splitter.addWidget(self.video_player)
        
        # Create visualization widget (right panel)
        # Right side: Stacked widget
        self.right_widget = QStackedWidget()
        splitter.addWidget(self.right_widget)

        self.visualization = VisualizationWidget(self.data_manager)  # Your plots
        self.info_panel = InfoPanel()
        self.info_panel.data_manager = self.data_manager

        self.right_widget.addWidget(self.visualization)  # index 0
        self.right_widget.addWidget(self.info_panel)     # index 1

        self.right_widget.setCurrentIndex(0)
        
        # self.visualization.data_manager = self.data_manager
        

        # Create timeline controller at bottom
        self.timeline = TimelineController()
        main_layout.addWidget(self.timeline)
        
        # Set initial splitter sizes (50% each)
        splitter.setSizes([600, 600])
        
        # Connect signals between components
        self.timeline.position_changed.connect(self.on_timeline_position_changed)
        self.timeline.position_changed.connect(self.visualization.update_decision_marker)
        self.video_player.frame_changed.connect(self.on_video_frame_changed)

        # Setup menu actions
        self.setup_menu()
        
        # Load the most recent data by default
        # self.load_latest_data()
        self.open_random_files()

    def toggle_view(self):
        if self.showing_info_panel:
            self.right_widget.setCurrentIndex(0)  # Show plots
            self.showing_info_panel = False
        else:
            self.right_widget.setCurrentIndex(1)  # Show info panel
            self.showing_info_panel = True



    
    def find_all_csv_files(self):
        """Find all CSV files in both logs and results directories"""
        csv_files = []
        
        # Check logs directory
        if os.path.exists(DEFAULT_LOG_DIR):
            log_csv_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.csv')]
            csv_files.extend([(os.join(DEFAULT_LOG_DIR, f), 'log') for f in log_csv_files])
        
        # Check results directory
        if os.path.exists(RESULTS_LOG_DIR):
            # Recursively walk through results directory structure
            for root, dirs, files in os.walk(RESULTS_LOG_DIR):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append((os.path.join(root, file), 'result'))
        
        return csv_files

    def open_random_files(self):
        """Randomly select a log file and its corresponding video file."""
        # List all CSV log files in the logs directory
        csv_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.csv')]
        if not csv_files:
            print("No log files found.")
            return

        # Pick a random csv file
        random_csv = random.choice(csv_files)
        csv_path = os.path.join(DEFAULT_LOG_DIR, random_csv)

        # Try to find the corresponding video file (same base name, .mp4)
        base_name = os.path.splitext(random_csv)[0]
        video_path = os.path.join(DEFAULT_LOG_DIR, f"{base_name}.mp4")

        # If the video doesn't exist, pick a random video file
        if not os.path.exists(video_path):
            video_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.mp4')]
            if not video_files:
                print("No video files found.")
                return
            video_path = os.path.join(DEFAULT_LOG_DIR, random.choice(video_files))

        # Load the selected files
        self.load_data(csv_path, video_path)

    def on_timeline_position_changed(self, position):
        """Handle timeline position changes (0-100%)"""
        # Update video position
        self.video_player.seek_percent(position)
        
        # Calculate corresponding step for visualization
        frame = self.video_player.current_frame
        step = self.timeline.frame_to_step(frame)
        
        # Update visualization position without triggering back-propagation
        self.visualization.update_position(step, from_timeline=True)

        self.info_panel.update_state(step)

    def setup_menu(self):
        """Create the application menu bar with actions"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Open random video and log
        random_action = file_menu.addAction('Open Random Log and Video')
        random_action.triggered.connect(self.open_random_files)
        
        # Open submenu for results
        results_menu = file_menu.addMenu('Open from Results')
        
        # Action to browse results directory
        browse_results = results_menu.addAction('Browse Results Directory...')
        browse_results.triggered.connect(self.browse_results_directory)
        
        # Add quick access to recent episodes if they exist
        if os.path.exists(RESULTS_LOG_DIR):
            self.populate_recent_episodes_menu(results_menu)
        
        # Open logs action
        open_logs_action = file_menu.addAction('Open from Logs Directory...')
        open_logs_action.triggered.connect(self.open_log_files)
        
        # Exit action
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Toggle visualization types
        show_cumulative = view_menu.addAction('Show Cumulative Rewards')
        show_cumulative.setCheckable(True)
        show_cumulative.setChecked(True)
        show_cumulative.triggered.connect(lambda checked: self.visualization.toggle_view('cumulative', checked))
        
        show_components = view_menu.addAction('Show Reward Components')
        show_components.setCheckable(True)
        show_components.setChecked(True)
        show_components.triggered.connect(lambda checked: self.visualization.toggle_view('components', checked))

        show_decision = view_menu.addAction('Show Decision Attribution')
        show_decision.setCheckable(True)
        show_decision.setChecked(True)
        show_decision.triggered.connect(lambda checked: self.visualization.toggle_view('decision', checked))



    def populate_recent_episodes_menu(self, menu):
        """Add menu items for recent episodes in the results directory"""
        # Look for dreamer_v2 directory
        dreamer_dir = os.path.join(RESULTS_LOG_DIR, 'dreamer_v2')
        if not os.path.exists(dreamer_dir):
            return
            
        # Look for checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(dreamer_dir) 
                        if os.path.isdir(os.path.join(dreamer_dir, d)) 
                        and d.startswith('checkpoint_')]
        
        if not checkpoint_dirs:
            return
        
        # Sort by checkpoint number (descending)
        checkpoint_dirs.sort(reverse=True)
        
        # For the most recent checkpoint, add episode entries
        recent_checkpoint = checkpoint_dirs[0]
        checkpoint_path = os.path.join(dreamer_dir, recent_checkpoint)
        
        # Add a submenu for this checkpoint
        checkpoint_menu = menu.addMenu(f"Recent: {recent_checkpoint}")
        
        # Find episode directories
        episode_dirs = [d for d in os.listdir(checkpoint_path)
                    if os.path.isdir(os.path.join(checkpoint_path, d))
                    and d.startswith('episode_')]
        
        # Sort by episode number
        episode_dirs.sort()
        
        # Add the most recent 5 episodes
        for episode_dir in episode_dirs[-5:]:
            episode_path = os.path.join(checkpoint_path, episode_dir)
            
            # Look for data.csv in this episode
            data_path = os.path.join(episode_path, 'data.csv')
            if os.path.exists(data_path):
                # Add menu item for this episode
                episode_action = checkpoint_menu.addAction(f"{episode_dir}")
                # Use lambda with default arg to capture current value
                episode_action.triggered.connect(
                    lambda checked, path=data_path: self.load_episode(path)
                )

    # VisMain.py  –  inside load_episode()
    def load_episode(self, csv_path):
        dir_path = os.path.dirname(csv_path)

        # NEW: take *any* mp4 in this folder
        video_files = [f for f in os.listdir(dir_path) if f.endswith(".mp4")]
        video_path  = os.path.join(dir_path, video_files[0]) if video_files else None

        if not video_path:               # still nothing? ask user
            video_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", dir_path, "Video Files (*.mp4)"
            )
            if not video_path:
                return

        self.load_data(csv_path, video_path)


    def browse_results_directory(self):
        """Browse the results directory structure to find CSV files"""
        if not os.path.exists(RESULTS_LOG_DIR):
            print("Results directory does not exist.")
            return
            
        # Use directory dialog to browse the complex structure
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Episode Directory", RESULTS_LOG_DIR
        )
        
        if not dir_path:
            return  # User cancelled
            
        # Look for data.csv in this directory
        csv_path = os.path.join(dir_path, 'data.csv')
        if not os.path.exists(csv_path):
            # Try to find any CSV
            csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(dir_path, csv_files[0])
            else:
                print(f"No CSV files found in {dir_path}")
                return
        
        # Find video in the same directory
        video_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
        
        video_path = None
        if video_files:
            # Take the first video file
            video_path = os.path.join(dir_path, video_files[0])
        
        # If no video found, look for video elsewhere
        if not video_path:
            video_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", dir_path, "Video Files (*.mp4)"
            )
            if not video_path:
                return  # User cancelled
        
        # Load the data
        self.load_data(csv_path, video_path)

    def open_log_files(self):
        """Open dialog to select files from logs directory"""
        log_file, _ = QFileDialog.getOpenFileName(
            self, "Select Event Log File", DEFAULT_LOG_DIR, "CSV Files (*.csv)"
        )
        
        if log_file:
            self._handle_file_selection(log_file)

    def _handle_file_selection(self, log_file):
        """Handle file selection from either directory"""
        # Try to find corresponding video with same base name
        base_name = os.path.splitext(os.path.basename(log_file))[0]
        video_dir = os.path.dirname(log_file)
        possible_video = os.path.join(video_dir, f"{base_name}.mp4")
        video_file = possible_video if os.path.exists(possible_video) else None
        
        # If no matching video found, ask user to select
        if not video_file:
            video_file, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", video_dir, "Video Files (*.mp4)"
            )
            
        if video_file:
            self.load_data(log_file, video_file)
    
    def open_files(self):
        """Open dialog to select log and video files"""
        
        log_file, _ = QFileDialog.getOpenFileName(
            self, "Select Event Log File", DEFAULT_LOG_DIR, "CSV Files (*.csv)"
        )
        
        if log_file:
            # Try to find corresponding video with same base name
            base_name = os.path.splitext(os.path.basename(log_file))[0]
            video_dir = os.path.dirname(log_file)
            possible_video = os.path.join(video_dir, f"{base_name}.mp4")
            
            video_file = possible_video if os.path.exists(possible_video) else None
            
            # If no matching video found, ask user to select
            if not video_file:
                video_file, _ = QFileDialog.getOpenFileName(
                    self, "Select Video File", video_dir, "Video Files (*.mp4)"
                )
            
            if video_file:
                self.load_data(log_file, video_file)
    
    def load_latest_data(self):
        """Find and load the most recent log and video files"""
        
        try:
            # Find CSV files in log directory
            csv_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.csv')]
            
            if not csv_files:
                return
            
            # Sort by modification time (newest first)
            csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(DEFAULT_LOG_DIR, x)), reverse=True)
            latest_csv = os.path.join(DEFAULT_LOG_DIR, csv_files[0])
            
            # Find video files in log directory
            video_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.mp4')]
            
            if not video_files:
                return
                
            # Sort by modification time (newest first)
            video_files.sort(key=lambda x: os.path.getmtime(os.path.join(DEFAULT_LOG_DIR, x)), reverse=True)
            latest_video = os.path.join(DEFAULT_LOG_DIR, video_files[0])
            
            self.load_data(latest_csv, latest_video)
            
        except Exception as e:
            print(f"Error loading latest data: {e}")
    
    def load_data(self, log_file, video_file):
        """Load data from log and video files"""

        if not self.data_manager.load_data(log_file):
            print(f"Failed to load data from {log_file}")
            return False
    
        # Load event data
        self.data_manager.load_data(log_file)
        
                
        # Pass data to visualization
        self.visualization.set_data(
            self.data_manager.time_steps,
            self.data_manager.reward_log,
            self.data_manager.action_log,  # Use action IDs directly
            self.data_manager.reward_components
        )

        # Build / refresh the decision-attribution comparison plot
        self.visualization.rebuild_decision_plot()

        
        # Pass video to player
        self.video_player.load_video(video_file)

        
        # Setup timeline controller
        total_steps = len(self.data_manager.time_steps)
        total_frames = self.video_player.total_frames
        self.timeline.setup(total_steps, total_frames)
        
        # Update window title
        self.setWindowTitle(f"Crafter Analysis - {os.path.basename(log_file)}")
    
    def on_timeline_position_changed(self, position):
        """Handle timeline position changes (0-100%)"""
        
        # Update video position
        self.video_player.seek_percent(position)
        
        # Calculate corresponding step for visualization
        frame = self.video_player.current_frame
        step = self.timeline.frame_to_step(frame)
        
        # Update visualization position without triggering back-propagation
        self.visualization.update_position(step, from_timeline=True)
    
        # Define an offset for synchronization 
        self.frame_offset = 1

    # Update the on_video_frame_changed method
    def on_video_frame_changed(self, frame):
        """Handle video frame changes"""

         # Avoid processing the same frame repeatedly at video end
        if hasattr(self, 'last_processed_frame') and self.last_processed_frame == frame:
            return
        self.last_processed_frame = frame

        # Update timeline position
        position = (frame / self.video_player.total_frames) * 100
        self.timeline.set_position(position, from_video=True)
        
        # Calculate corresponding step for visualization with offset
        step = self.timeline.frame_to_step(frame, offset=self.frame_offset)
        
        # Update visualization position
        self.visualization.update_position(step, from_video=True)

        self.info_panel.update_state(step) 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


