<?php
class block_intelligent_tutor extends block_base {
    public function init() {
        $this->title = get_string('intelligent_tutor', 'block_intelligent_tutor');
    }

    public function get_content() {
        if ($this->content !== null) {
          return $this->content;
        }

        $this->content = new stdClass;
        
        // This is where your FastAPI server will run locally
        $fastapi_url = "http://localhost:8000/";
        
        // Pass Moodle user's first name to the chatbot if logged in
        global $USER;
        $username = isset($USER->firstname) ? $USER->firstname : 'Student';

        // Display an iframe pointing to our FastAPI static frontend
        $this->content->text   = '<div style="height: 500px; width: 100%; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">';
        $this->content->text  .= '<iframe src="' . $fastapi_url . '?user=' . urlencode($username) . '" width="100%" height="100%" frameborder="0"></iframe>';
        $this->content->text  .= '</div>';
        
        return $this->content;
    }
}
