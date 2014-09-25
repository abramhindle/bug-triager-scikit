# Convert github to a json format that I like.
#
# I am licensing my contributions under the MIT License but
# the script was based on API examples from:
#
# Original source of octokit example:
#   (C) 2012 Henare Degan henare http://github.com/henare
#   https://gist.github.com/henare/1106008
#   (C) 2013 Tod Karpinski tkarpinski
#   https://gist.github.com/tkarpinski/2369729
#
# Whether or not this is a derived work or not is up to them, but my contribution has the following
# license
#
# Copyright (C) 2013 Abram Hindle
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


require 'octokit'
require 'csv'
require 'date'
require 'json'

# Github credentials to access your private project
config = JSON.load(File.new("config.json"))
USERNAME=ENV["GHUSERNAME"] || config["GHUSERNAME"]
PASSWORD=ENV["GHPASSWORD"] || config["GHPASSWORD"]

# The project you are mirroring issues from
USER=ENV["GHUSER"] || config["GHUSER"]
PROJECT=ENV["GHPROJECT"] || config["GHPROJECT"]

REPO = "#{USER}/#{PROJECT}"



puts "Getting issues from Github..."

class GH
  attr_accessor :repo
  attr_accessor :client
  attr_accessor :issues
  attr_accessor :comments

  def initialize(repo)
    @repo = repo
    @client = Octokit::Client.new(:login => USERNAME, :password => PASSWORD)
    @comments = {}
    #@comments.default = []
  end

  def get_issues()
    repo = @repo
    client = @client
    puts(repo)
    temp_issues = []
    issues = []
    page = 0
    begin
      page = page + 1
      puts "Closed Issues Page #{page}"
      temp_issues = client.list_issues(repo, :state => "closed", :page => page)
      issues = issues + temp_issues;
    end while not temp_issues.empty?
    temp_issues = [] 
    page = 0
    begin
      page = page + 1
      puts "Open Issues Page #{page}"
      temp_issues = client.list_issues(repo, :state => "open", :page => page)
      issues = issues + temp_issues;
    end while not temp_issues.empty?
    issues = issues.map { |issue|
                          r = issue.to_hash() 
			  r[:user] = r[:user].to_hash
			  r[:assignee] = r[:assignee] ? r[:assignee].to_hash : nil
                          r[:pull_request] = r[:pull_request] ? r[:pull_request].to_hash : nil
			  r[:labels] = r[:labels].map { | label | 
                                                        label.to_hash }
			  r }
    puts issues
    @issues = issues
    return issues
  end
  # get them one at a time

  def get_comments()
    return self.get_all_comments()
    #issues = @issues
    #issues.each do |issue|
    #  puts "Processing issue #{issue['number']}..."
    #  comments = client.issue_comments( REPO, issue.number )      
    #  num = issue['number']
    #  @comments[num] = comments
    #  issue["comments"] = comments
    #end
    #return @comments
  end

  # do the github work
  def _get_comments_from_gh()
    comments = []
    page = 1
    done = false
    until done
      puts "Comment Page #{page}"
      newcomments = self.client.issues_comments( REPO, { :page => page} )
      comments += newcomments
      done = newcomments == []
      page = page + 1
    end
    page = 1
    done = false
    until done
      puts "PR Comment Page #{page}"
      newcomments = self.client.pull_requests_comments( REPO, { :page => page} )
      comments += newcomments
      done = newcomments == []
      page = page + 1
    end
    comments = comments.map { |comment| 
                              r = comment.to_hash
			      r[:user] = r[:user] ? r[:user].to_hash : ""
			      r }
    return comments
  end

  # munge it all
  def get_all_comments()

    comments = self._get_comments_from_gh()

    comments.each do |comment|
      if (comment[:issue_url])
        comment["issue_id"] = comment[:issue_url].split("/")[-1].to_i
      else
        comment["issue_id"] = comment[:pull_request_url].split("/")[-1].to_i
      end
      num = comment["issue_id"]
      if (!@comments[num])
        @comments[num] = []
      end
      @comments[num] = @comments[num] + [ comment ]
    end
    issues = @issues
    issues.each do |issue|
      mynum = issue[:number]
      if (!@comments[mynum])
        @comments[mynum] = []
      end
      issue["comments"] = @comments[mynum]
    end
    return @comments
  end

  def generate_large_json()
    # convert issues and comments to json with appropriate fields
    # need owner, reportedBy title description content
    # number -> "_id"
    # user->login to reportedBy
    # assignee->login to reportedBy
    # title to title
    # body -> content
    # comments need author and content    
    # so map body to content
    # user->login to author
    # body to content

    issues = @issues.map { |issue| 
      newissue = Hash.new
      newissue["_id"] = issue[:number]
      newissue["created_at"] = issue[:created_at]
      newissue["reportedBy"] = issue[:user][:login]
      newissue["owner"] = ((issue[:assignee])?issue[:assignee][:login]:"")
      newissue["content"] = (issue[:title] ? issue[:title] : "") + "\n" + (issue[:body] ? issue[:body] : "")
#       puts(issue["comments"].length.to_s)
      newissue["comments"] = issue["comments"].map { |comment| 
        newcomment = Hash.new
        newcomment["content"] = comment[:body]
        newcomment["author"] = comment[:user].is_a?(Hash) ? comment[:user][:login] : ""
	newcomment
      }
      { "doc" => newissue }
    }
    return JSON.pretty_generate( {
                                   "rows" => issues
                                 })
  end
end

gh = GH.new(REPO);
issues = gh.get_issues();
comments = gh.get_comments();
1
issues_json = JSON.pretty_generate( issues )
File.new("issues.json","w").write( issues_json )
comments_json = JSON.pretty_generate( comments )
File.new("comments.json","w").write( comments_json )
issues_json = JSON.pretty_generate( gh.issues )
File.new("issues.json","w").write( issues_json )
File.new("large.json","w").write( gh.generate_large_json )

